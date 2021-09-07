'''
File: Problem.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 01:42
Last Modified: 2021-06-11 15:42
--------------------------------------------
Description:
'''
import sys
import warnings
sys.path.append("../")
from abc import ABC, abstractclassmethod
import numpy as np
from numba import jit
import src.utils as utils
import src.natOG.printUtils as printUtils

class Problem(ABC):
    def __init__(self, f, r) -> None:
        super().__init__()
        self.f = f
        self.r = r

    @abstractclassmethod
    def funcf(self, x):
        """
        function evaluation
        """
        pass

    @abstractclassmethod
    def funcr(self, x):
        """
        function evaluation
        """
        pass

    @abstractclassmethod
    def gradf(self, x):
        """
        gradient evaluation
        """
        pass

    @abstractclassmethod
    def ipg(self, x, gradfx, alpha):
        """
        return an inexact proximal gradient point
        """
        pass


class ProbNatOG(Problem):
    def __init__(self, f, r) -> None:
        super().__init__(f, r)
        self.K = self.r.K
        self.n, self.p = self.f.n, self.f.p
        self.starts = self.r.starts
        self.ends = self.r.ends
        self.Lambda_group = self.r.Lambda_group
        self.full_group_index = np.arange(self.K)
        self.e = np.ones((self.K, 1))

    def funcf(self, x):
        return self.f.evaluate_function_value(x)

    def funcr(self, x):
        return self.r.func(x)

    def gradf(self, x):
        # least square and logit loss doesn't require this
        return self.f.gradient()

    def ipg(self, xk, gradfxk, alphak, params, outter_iter, Y_init=None, outID=None):
        """
            inexactly solve the proximal graident problem
        """
        if params['subsolver'] == 'projectedGD':
            return self._projectedGD(xk, gradfxk, alphak, params, outter_iter, Y_init, outID)
        else:
            raise ValueError(f"Unknown subsolver:{params['subsolver']}.")
    

    
    def _xFromY(self, Y, uk, alphak):
        diff = uk - alphak * np.sum(Y, axis=1, keepdims=True)
        ukabs = np.abs(uk)
        pos_flag, neg_flag = (diff > ukabs).reshape(-1), (diff < -ukabs).reshape(-1)
        x = diff + 0.0
        x[pos_flag] = ukabs[pos_flag]
        x[neg_flag] = -ukabs[neg_flag]
        return x

    def _proj(self, Y):
        """
            perform projection of Y such that
            ||Yg|| <= alphak * wg
        """
        return _proj_jit(Y, self.K, self.starts, self.ends, self.Lambda_group)

    def _dual(self, Y, x, uk, alphak):
        """
            w(x, Y) = - (1/(2alphak) ||x-uk||^2 + x^TYe)
        """
        diff = x-uk
        part1 = np.sum(diff*diff) / (2*alphak)
        part2 = np.sum(x * np.sum(Y, axis=1, keepdims=True))
        dual = -(part1+part2)
        return dual

    
    def _projectedGD(self, xk, gradfxk, alphak, params, outter_iter, Y_init=None, outID=None):
        """
            perform projected gradient descent to solve the 
                min 
            termination type:
                params['inexact_type'] 1: ck||s_k||^2
                params['inexact_type'] 2: gamma_2
                params['inexact_type'] 3: O(1/k^3)
        reference: https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
        The dual variable Y, which is a sparse p-by-K matrix. In stead of storing the whole
        Matrix, we operate on its compressed form.

        """
        uk = xk - alphak * gradfxk       
        # Initialize dual variable Y
        if Y_init is None:
            Y = np.zeros((self.p, self.K))
        else:
            Y = Y_init
        self.dualProbDim = np.sum(self.ends - self.starts)
        # process Y to make sure its feasibility
        Y = self._proj(Y)
        if params['subsolver_verbose']:
            printUtils.print_subsolver_header(self.dualProbDim, params['subsolver'],
                                              params['inexact_type'], outter_iter, outID)
        # ----------------- perform the projected gradient descent -------------
        x = self._xFromY(Y, uk, alphak)
        fdualY = self._dual(Y, x, uk, alphak)
        iters = 0
        while True:
            # ----------------------- one iteration of PGD -------------------------
            gradY = -x@(self.e).T
            Ytrial = self._proj(Y - params['projectedGD']['stepsize'] * gradY)
            norm_grad = utils.l2_norm(gradY.reshape(-1,1))
            # search direction
            d = Ytrial - Y
            #  ------------------------ backtrack line-search ------------------
            bak = 0
            stepsize = 1.0
            dirder = np.sum(gradY * d)
            while True:
                Ytrial = Y + stepsize * d
                xtrial = self._xFromY(Ytrial, uk, alphak)
                fdualYtrial = self._dual(Ytrial, xtrial, uk, alphak)
                lhs =  fdualYtrial - fdualY 
                rhs = dirder * stepsize * params['eta'] 
                print(f"iter:{iters:2d} | bak:{bak:2d} | lhs:{lhs:3.3e} | rhs:{rhs:3.3e} | dirder:{dirder:3.3e}")
                if lhs <= rhs:
                    break
                if (np.abs(dirder) <= 1e-15) or (bak == 0 and np.abs(lhs) <= 1e-15) or  (bak == 0 and np.abs(rhs) <= 1e-15):
                    break
                if bak > 50:
                    flag = 'lscfail'
                    warnings.warn("Linesearch failed in projected GD")
                    print(f"lhs:{lhs:3.3e} | rhs:{rhs:3.3e} | dirder:{dirder:3.3e}")
                    return x, Y, flag, iters, -999, -999
                bak += 1
                stepsize *= params['xi']
            norm_d = utils.l2_norm(d) * stepsize
            # ----------------------- check termination -------------------------------------
            if bak == 0:
                params['projectedGD']['stepsize'] = min(1e8, params['projectedGD']['stepsize']*1.1)
            else:
                params['projectedGD']['stepsize'] = max(1e-8, params['projectedGD']['stepsize']*0.9)
            # case I: max number of iterations reached
            if iters == params['projectedGD']['maxiter']:
                flag = 'maxiters'
                warnings.warn("inexactness conditon is definitely not satisfied due to the maxiter limit.")
                break
            # case II: desidered termination conditon
            # check duality gap
            diff = xtrial-uk
            phipxkplus1 = np.sum(diff*diff) / (2*alphak) + self.r.func(xtrial)
            phipLB = -fdualYtrial
            # fdualYtrial is filpped for minimization
            gap = phipxkplus1  - phipLB
            if params['inexact_type'] == 1:
                sk = xtrial - xk
                self.ck = (np.sqrt(6 / ((1 + params['gamma1']) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
                epsilon = self.ck * (np.dot(sk.T, sk)[0][0])
            elif params['inexact_type'] == 2:
                diff = xk-uk
                phipxk = np.sum(diff*diff) / (2*alphak) + self.r.func(xk)
                epsilon = (1-params['gamma2']) * (phipxk - phipLB)
                # print(f"Scaled phipxk:{phipxk_s:3.3e} | Scaled phipxk+1:{phipxtrial_s:3.3e} | Scaled phipLB:{phipLB_s:3.3e}")
            elif params['inexact_type'] == 3:
                epsilon = params['schimdt_const'] / (outter_iter ** params['delta'])
            else:
                raise ValueError(f" inextact_type:{params['inexact_type']} is not implemeted!")
            inexact_cond = (gap <= epsilon)
            if params['subsolver_verbose']:
                printUtils.print_subsolver_iterates(iters, norm_grad, params['projectedGD']['stepsize'], phipxkplus1, phipLB, gap,
                                                    bak, stepsize, norm_d, outID)
            Y = Ytrial 
            x = xtrial 
            fdualY = fdualYtrial
            iters += 1
            if inexact_cond:
                flag = 'desired '
                break
            # if gap <= 1e-10:
            #     flag = 'desired '
            #     break
        return x, Y, flag, iters, gap, epsilon


# @jit(nopython=True, cache=True)
def _proj_jit(Y, K, starts, ends, Lambda_group):
    """
        perform projection of Y such that
        ||Yg|| <= wg
    """
    for i in range(K):
        start, end = starts[i], ends[i]
        Yg = Y[start:end, [i]]
        norm_Yg = utils.l2_norm(Yg)
        wg = Lambda_group[i]
        if norm_Yg > wg:
            Y[start:end, [i]] = (wg / norm_Yg) * Yg
    return Y

