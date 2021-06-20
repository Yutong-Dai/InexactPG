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
        self.M = np.sum(self.Lambda_group)
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
    

    
    def _zFromY(self, Y, uk, nonZeroGroupFlag):
        return _zFromY_jit(Y, uk, nonZeroGroupFlag, 
                       self.K, self.starts, self.ends, self.r.Ystarts, self.r.Yends)

    def _proj(self, Y, x, stepsize, alphak, nonZeroGroupFlag):
        """
            perform projection of Y such that
            ||Yg|| <= alphak * wg
        """
        return _proj_jit(Y, x, stepsize, alphak, nonZeroGroupFlag, 
          self.K, self.starts, self.ends, self.r.Ystarts, self.r.Yends, self.Lambda_group)

    def _dual(self, Y, x, uk, nonZeroGroupFlag):
        return _dual_jit(Y, x, uk, nonZeroGroupFlag,
          self.K, self.starts, self.ends, self.r.Ystarts, self.r.Yends)

    
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
        # phipxk = 0.5 * utils.l2_norm(xk-uk_origin) + alphak * self.r.func(xk)
        # ----------------------- pre-screening to select variables to work on ---------
        nonZeroGroupFlag, entrySignFlag, vk = _identifyZeroGroups(uk, alphak, 
                                                                    self.Lambda_group, 
                                                                    self.starts, self.ends, self.K)
                                                                   
        # print(np.sum(xk_s<0))
        # Initialize dual variable Y
        if Y_init is None:
            Y = np.zeros((self.r.Yends[-1], 1))
        else:
            Y = Y_init
        self.dualProbDim = 0
        # process Y to make sure its feasibility
        # also Y_{ij} = 0 if z_{i}= 0
        for i in range(self.K):
            Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
            if nonZeroGroupFlag[i]:
                self.dualProbDim += Yend - Ystart
                start, end = self.starts[i], self.ends[i]
                vkg = vk[start:end]
                Yg = Y[Ystart:Yend]
                Yg[vkg == 0] = 0.0
                norm_Yg = utils.l2_norm(Yg)
                radiusg = alphak * self.Lambda_group[i]
                if norm_Yg > radiusg:
                    Y[Ystart:Yend] = (radiusg / norm_Yg) * Yg
            else:
                Y[Ystart:Yend] = 0.0
        if params['subsolver_verbose']:
            printUtils.print_subsolver_header(self.dualProbDim, params['subsolver'],
                                              params['inexact_type'], outter_iter, outID)
        # ----------------- perform the projected gradient descent -------------
        z = self._zFromY(Y, vk, nonZeroGroupFlag)
        fdualY, part2 = self._dual(Y, z, vk, nonZeroGroupFlag)
        iters = 0
        while True:
            # ----------------------- one iteration of PGD -------------------------


            #  ------------------------ backtrack line-search ------------------
            bak = 0
            while True:
                Ytrial, gradY = self._proj(Y, z, params['projectedGD']['stepsize'], alphak, nonZeroGroupFlag)
                norm_grad = utils.l2_norm(gradY)
                ztrial = self._zFromY(Ytrial, vk, nonZeroGroupFlag)
                fdualYtrial, part2 = self._dual(Ytrial, ztrial, vk, nonZeroGroupFlag)
                d = Ytrial - Y
                norm_d = utils.l2_norm(d) 
                lhs =  fdualYtrial - fdualY - np.sum(gradY * (Ytrial - Y))
                rhs = (1 / (2 * params['projectedGD']['stepsize'])) * norm_d**2
                # print(f"iter:{iters:2d} | bak:{bak:2d} | lhs:{lhs:3.3e} | rhs:{rhs:3.3e} | dirder:{dirder:3.3e}")
                if lhs <= rhs:
                    ratio= rhs / (lhs+1e-16)
                    if ratio > 5:
                        params['projectedGD']['stepsize'] *= 1.25
                    break
                else:
                    ratio= lhs / (rhs*params['projectedGD']['stepsize'] + 1e-16)
                    if (2/params['projectedGD']['stepsize'] <= ratio):
                        params['projectedGD']['stepsize'] = 1 / ratio
                    else:
                        params['projectedGD']['stepsize'] *= 0.5
                    if (1/params['projectedGD']['stepsize'] > 2*self.K**2):
                        break
                bak += 1
            # ----------------------- check termination -------------------------------------
            # case I: max number of iterations reached
            if iters == params['projectedGD']['maxiter']:
                flag = 'maxiters'
                warnings.warn("inexactness conditon is definitely not satisfied due to the maxiter limit.")
                break
            # case II: desidered termination conditon
            # check duality gap
            xkplus1 = self._xfromz(ztrial, entrySignFlag)
            # phipxkhatplus1 = 0.5 * utils.l2_norm(xkplus1-uk_origin) + alphak * self.r.func(xkplus1)
            # phipLB = -_dual_jit(Ytrial, xkplus1, uk_origin, nonZeroGroupFlag, 
            #                     self.K, self.starts, self.ends, self.r.Ystarts, self.r.Yends)
            rztrail = self.r.func(ztrial) * alphak
            gap = rztrail - part2
            if params['inexact_type'] == 1:
                sk = xkplus1 - xk
                self.ck = (np.sqrt(6 / ((1 + params['gamma1']) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
                epsilon = self.ck * (np.dot(sk.T, sk)[0][0])
                aa = 0.5
                bb = utils.l2_norm(xkplus1 - uk) + alphak * self.M 
            elif params['inexact_type'] == 2:
                # epsilon = (1-params['gamma2']) * (phipxk - phipLB)
                pass
            elif params['inexact_type'] == 3:
                epsilon = params['schimdt_const'] / (outter_iter ** params['delta'])
                bb = utils.l2_norm(xkplus1 - uk) + alphak * self.M 
                aa = 0.5
            else:
                raise ValueError(f" inextact_type:{params['inexact_type']} is not implemeted!")
 
            Y = Ytrial 
            z = ztrial 
            fdualY = fdualYtrial
            iters += 1
            theta = (-bb + np.sqrt(bb**2 + 4 * aa * epsilon)) / (2 * aa)
            # theta_s = 0.5*(theta**2)
            theta_s = 0.5*theta
            # print(f"theta:{theta:3.5e} | theta^2:{theta**2:3.5e} | epsilon:{epsilon:3.5e}")
            inexact_cond = (gap <= theta_s)

            if params['subsolver_verbose']:
                printUtils.print_subsolver_iterates(iters, norm_grad, params['projectedGD']['stepsize'], 
                                                    rztrail, part2, gap, theta_s,
                                                    bak, 1.0, norm_d, outID)
            if inexact_cond:
                flag = 'desired '
                break
            # if gap <= 1e-10:
            #     flag = 'desired '
            #     break
        # # correct the sign for x
        # negx = (entrySignFlag == -1)
        # x[negx] = -x[negx]
        return xkplus1, Y, flag, iters, gap, epsilon, theta_s


    def _xfromz(self, z, entrySignFlag):
        # correct the sign for x
        x = z + 0.0
        negx = (entrySignFlag == -1)
        x[negx] = -x[negx]
        return x

    

@jit(nopython=True, cache=True)
def _identifyZeroGroups(uk, alphak, Lambda_group, starts, ends, K):
    # flip the sign of uk, such that vk>=0
    entrySignFlag = np.zeros_like(uk, dtype=np.int8)
    pos_uk, neg_uk = (uk >0).reshape(-1),  (uk<0).reshape(-1)
    entrySignFlag[pos_uk] = 1
    entrySignFlag[neg_uk] = -1
    vk = uk + 0.0
    vk[neg_uk] = -uk[neg_uk]
    # identify some zero-groups
    iterstep = 0
    nonZeroGroupFlag = np.full(K, True)
    while True:
        iterstep += 1
        if iterstep > K + 1:
            print("The code might have bugs.")
        newZeroGroup = 0
        for i in range(K):
            if nonZeroGroupFlag[i]:
                start, end = starts[i], ends[i]
                vkg = vk[start:end]
                vkg_norm = np.sqrt(np.sum(vkg * vkg))
                if vkg_norm <= alphak * Lambda_group[i]:
                    nonZeroGroupFlag[i]= False
                    vk[start:end] = 0.0
                    newZeroGroup += 1
        if newZeroGroup == 0:
            break
    return nonZeroGroupFlag, entrySignFlag, vk

@jit(nopython=True, cache=True)
def _zFromY_jit(Y, vk, nonZeroGroupFlag, K, starts, ends, Ystarts, Yends):
        """
            x = max( v - Y * e, 0)
            gradY
        """
        z = vk + 0.0
        for i in range(K):
            if nonZeroGroupFlag[i]:
                start, end = starts[i], ends[i]
                Ystart, Yend = Ystarts[i], Yends[i]
                z[start:end] -= Y[Ystart:Yend]
        znegFlag = (z<0).reshape(-1)
        z[znegFlag] = 0.0
        return z
@jit(nopython=True, cache=True)
def _proj_jit(Y, z, stepsize, alphak, nonZeroGroupFlag, 
          K, starts, ends, Ystarts, Yends, Lambda_group):
    """
        perform projection of Y such that
        ||Yg|| <= alphak * wg
    """
    Ynew = Y + 0.0
    gradY = np.zeros_like(Y)
    for i in range(K):
        if nonZeroGroupFlag[i]:
            Ystart, Yend = Ystarts[i], Yends[i]
            start, end = starts[i], ends[i]
            zg = z[start:end]
            gradY[Ystart:Yend] = - zg
            newYg = Y[Ystart:Yend] +  zg* stepsize
            norm_newYg = np.sqrt(np.sum(newYg*newYg))
            radiusg = alphak * Lambda_group[i]
            if norm_newYg > radiusg:
                Ynew[Ystart:Yend] = (radiusg / norm_newYg) * newYg
    return Ynew, gradY

@jit(nopython=True,cache=True)
def _dual_jit(Y, z, vk, nonZeroGroupFlag,
          K, starts, ends, Ystarts, Yends):
    """
        w(z, Y) = - (1/2 ||Y-vk||^2 + z^TYe)
    """
    diff = z-vk
    part1 = np.sum(diff*diff) / 2
    part2 = 0.0
    for i in range(K):
        if nonZeroGroupFlag[i]:
            Ystart, Yend = Ystarts[i], Yends[i]
            Yg = Y[Ystart:Yend]
            start, end = starts[i], ends[i]
            zg = z[start:end]
            part2 += np.sum(Yg * zg)
    dual = -(part1+part2)
    return dual, part2