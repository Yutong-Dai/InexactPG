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
    

    
    # def _xFromY(self, Y, uk, nonZeroGroupFlag):
    #     """
    #         x = max( u - Y * e, 0)
    #         gradY
    #     """
    #     x = uk + 0.0
    #     for i in range(self.K):
    #         if nonZeroGroupFlag[i]:
    #             start, end = self.starts[i], self.ends[i]
    #             Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
    #             x[start:end] -= Y[Ystart:Yend]
    #     x[x < 0] = 0.0
    #     return x
    def _xFromY(self, Y, uk, nonZeroGroupFlag):
        return _xFromY_jit(Y, uk, nonZeroGroupFlag, 
                       self.K, self.starts, self.ends, self.r.Ystarts, self.r.Yends)
    # def _proj(self, Y, x, stepsize, alphak, nonZeroGroupFlag):
    #     """
    #         perform projection of Y such that
    #         ||Yg|| <= alphak * wg
    #     """
    #     Ynew = Y + 0.0
    #     gradY = np.zeros_like(Y)
    #     for i in range(self.K):
    #         if nonZeroGroupFlag[i]:
    #             Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
    #             xstart, xend = self.starts[i], self.ends[i]
    #             xg = x[xstart:xend]
    #             gradY[Ystart:Yend] = - xg
    #             Yg = Y[Ystart:Yend] +  xg* stepsize
    #             norm_Yg = utils.l2_norm(Yg)
    #             radiusg = alphak * self.Lambda_group[i]
    #             if norm_Yg > radiusg:
    #                 Ynew[Ystart:Yend] = (radiusg / norm_Yg) * Yg
    #     return Ynew, gradY
    def _proj(self, Y, x, stepsize, alphak, nonZeroGroupFlag):
        """
            perform projection of Y such that
            ||Yg|| <= alphak * wg
        """
        return _proj_jit(Y, x, stepsize, alphak, nonZeroGroupFlag, 
          self.K, self.starts, self.ends, self.r.Ystarts, self.r.Yends, self.Lambda_group)

    # def _dual(self, Y, x, uk, zeroGroupFlag):
    #     """
    #         w(x, Y) = - (1/2 ||Y-uk||^2 + x^TYe)
    #     """
    #     part1 =  utils.l2_norm(x-uk) ** 2 / 2
    #     part2 = 0.0
    #     for i in range(self.K):
    #         if zeroGroupFlag[i]:
    #             Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
    #             Yg = Y[Ystart:Yend]
    #             start, end = self.starts[i], self.ends[i]
    #             xg = x[start:end]
    #             part2 += np.sum(Yg * xg)
    #     dual = -(part1+part2)
    #     return dual, part2
    def _dual(self, Y, x, uk, zeroGroupFlag):
        return _dual_jit(Y, x, uk, zeroGroupFlag,
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
        # ----------------------- pre-screening to select variables to work on ---------
        nonZeroGroupFlag, entrySignFlag, uk, phipxk_s, xk_s = _identifyZeroGroups(xk, uk, alphak, 
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
        # also Y_{ij} = 0 if x_{i}= 0
        for i in range(self.K):
            Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
            if nonZeroGroupFlag[i]:
                self.dualProbDim += Yend - Ystart
                start, end = self.starts[i], self.ends[i]
                ukg = uk[start:end]
                Yg = Y[Ystart:Yend]
                Yg[ukg == 0] = 0.0
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
        x = self._xFromY(Y, uk, nonZeroGroupFlag)
        fdualY, part2 = self._dual(Y, x, uk, nonZeroGroupFlag)
        iters = 0
        while True:
            # ----------------------- one iteration of PGD -------------------------
            Ytrial, gradY = self._proj(Y, x, params['projectedGD']['stepsize'], alphak, nonZeroGroupFlag)
            norm_grad = utils.l2_norm(gradY)
            # search direction
            d = Ytrial - Y
            #  ------------------------ backtrack line-search ------------------
            bak = 0
            stepsize = 1.0
            dirder = np.sum(gradY * d)
            while True:
                Ytrial = Y + stepsize * d
                xtrial = self._xFromY(Ytrial, uk, nonZeroGroupFlag)
                fdualYtrial, part2 = self._dual(Ytrial, xtrial, uk, nonZeroGroupFlag)
                lhs =  fdualYtrial - fdualY 
                rhs = dirder * stepsize * params['eta'] 
                # print(f"iter:{iters:2d} | bak:{bak:2d} | lhs:{lhs:3.3e} | rhs:{rhs:3.3e} | dirder:{dirder:3.3e}")
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
            rxtrail_s = self.r.func(xtrial) * alphak
            gap = rxtrail_s  - part2
            if params['inexact_type'] == 1:
                # sk = x - xk
                # sk = x - xk_s
                sk = xtrial - xk_s
                self.ck = (np.sqrt(6 / ((1 + params['gamma1']) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
                epsilon = self.ck * (np.dot(sk.T, sk)[0][0])
            elif params['inexact_type'] == 2:
                # phipxtrial_s = 0.5 * np.sum((xtrial-uk)*(xtrial-uk)) + rxtrail_s
                phipLB_s = -fdualYtrial
                epsilon = (1-params['gamma2']) * (phipxk_s - phipLB_s)
                # print(f"Scaled phipxk:{phipxk_s:3.3e} | Scaled phipxk+1:{phipxtrial_s:3.3e} | Scaled phipLB:{phipLB_s:3.3e}")
            elif params['inexact_type'] == 3:
                epsilon = params['schimdt_const'] / (outter_iter ** params['delta'])
            else:
                raise ValueError(f" inextact_type:{params['inexact_type']} is not implemeted!")
            inexact_cond = (gap <= epsilon)
            if params['subsolver_verbose']:
                printUtils.print_subsolver_iterates(iters, norm_grad, params['projectedGD']['stepsize'], rxtrail_s, part2, gap,
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
        # correct the sign for x
        negx = (entrySignFlag == -1)
        x[negx] = -x[negx]
        return x, Y, flag, iters, gap, epsilon

    

@jit(nopython=True, cache=True)
def _identifyZeroGroups(xk, uk, alphak, Lambda_group, starts, ends, K):
    # flip the sign of uk, such that uk>=0
    xk_s = xk + 0.0
    uk_old = uk + 0.0
    entrySignFlag = np.zeros_like(uk, dtype=np.int8)
    pos_uk, neg_uk = (uk >0).reshape(-1),  (uk<0).reshape(-1)
    entrySignFlag[pos_uk] = 1
    entrySignFlag[neg_uk] = -1
    uk[neg_uk] = -uk[neg_uk]
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
                ukg = uk[start:end]
                ukg_norm = np.sqrt(np.sum(ukg * ukg))
                if ukg_norm <= alphak * Lambda_group[i]:
                    nonZeroGroupFlag[i]= False
                    # nonZeroBlockFlag[start:end] = False
                    uk[start:end] = 0.0
                    newZeroGroup += 1
        if newZeroGroup == 0:
            break
    ukzeroFlag = (uk == 0.0).reshape(-1)
    xk_s[ukzeroFlag] = xk_s[ukzeroFlag] - uk_old[ukzeroFlag]
    xk_s[neg_uk] = -xk[neg_uk]
    rxk_s = 0.0
    xk_s = xk + 0.0
    xk_s[(xk<0).reshape(-1)] = -xk[(xk<0).reshape(-1)]
    for i in range(K):
        start, end = starts[i], ends[i]
        xk_sg = xk_s[start:end]
        rxk_s += Lambda_group[i] * np.sqrt(np.dot(xk_sg.T, xk_sg))[0][0]
    # translated phip(xk;xk,alphak)
    phipxk = 0.5 * np.sum((xk_s-uk)*(xk_s-uk)) + alphak * rxk_s
    return nonZeroGroupFlag, entrySignFlag, uk, phipxk, xk_s

@jit(nopython=True, cache=True)
def _xFromY_jit(Y, uk, nonZeroGroupFlag, K, starts, ends, Ystarts, Yends):
        """
            x = max( u - Y * e, 0)
            gradY
        """
        x = uk + 0.0
        for i in range(K):
            if nonZeroGroupFlag[i]:
                start, end = starts[i], ends[i]
                Ystart, Yend = Ystarts[i], Yends[i]
                x[start:end] -= Y[Ystart:Yend]
        xnegFlag = (x<0).reshape(-1)
        x[xnegFlag] = 0.0
        return x
@jit(nopython=True, cache=True)
def _proj_jit(Y, x, stepsize, alphak, nonZeroGroupFlag, 
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
            xstart, xend = starts[i], ends[i]
            xg = x[xstart:xend]
            gradY[Ystart:Yend] = - xg
            Yg = Y[Ystart:Yend] +  xg* stepsize
            norm_Yg = np.sqrt(np.sum(Yg*Yg))
            radiusg = alphak * Lambda_group[i]
            if norm_Yg > radiusg:
                Ynew[Ystart:Yend] = (radiusg / norm_Yg) * Yg
    return Ynew, gradY

@jit(nopython=True,cache=True)
def _dual_jit(Y, x, uk, zeroGroupFlag,
          K, starts, ends, Ystarts, Yends):
    """
        w(x, Y) = - (1/2 ||Y-uk||^2 + x^TYe)
    """
    diff = x-uk
    part1 = np.sum(diff*diff) / 2
    part2 = 0.0
    for i in range(K):
        if zeroGroupFlag[i]:
            Ystart, Yend = Ystarts[i], Yends[i]
            Yg = Y[Ystart:Yend]
            start, end = starts[i], ends[i]
            xg = x[start:end]
            part2 += np.sum(Yg * xg)
    dual = -(part1+part2)
    return dual, part2