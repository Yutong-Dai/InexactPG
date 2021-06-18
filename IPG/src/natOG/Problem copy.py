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
    #         y = u - Y * e - x
    #     """
    #     x = uk + 0.0
    #     for i in range(self.K):
    #         if nonZeroGroupFlag[i]:
    #             start, end = self.starts[i], self.ends[i]
    #             Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
    #             x[start:end] -= Y[Ystart:Yend]
    #     # distance to the boundary
    #     y = x + 0.0
    #     y[x >= 0] = 0
    #     x[x < 0] = 0
    #     return x,y
    def _xFromY(self, Y, uk, nonZeroGroupFlag):
        """
            x = max( u - Y * e, 0)
            gradY
        """
        x = uk + 0.0
        gradY = np.zeros_like(Y)
        for i in range(self.K):
            if nonZeroGroupFlag[i]:
                start, end = self.starts[i], self.ends[i]
                Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
                x[start:end] -= Y[Ystart:Yend]
                gradY[Ystart:Yend] = -x[start:end]
        x[x < 0] = 0.0
        return x, gradY

    def _proj(self, Y, alphak, nonZeroGroupFlag):
        """
            perform projection of Y such that
            ||Yg|| <= alphak * wg
        """
        for i in range(self.K):
            if nonZeroGroupFlag[i]:
                Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
                Yg = Y[Ystart:Yend]
                norm_Yg = utils.l2_norm(Yg)
                radiusg = alphak * self.Lambda_group[i]
                if norm_Yg > radiusg:
                    Y[Ystart:Yend] = (radiusg / norm_Yg) * Yg
        return Y

    def _dual(self, Y, x, uk):
        """
            w(x, Y) = - (1/2 ||Y-uk||^2 + x^TYe)
        """
        part1 =  utils.l2_norm(x-uk) ** 2 / 2
        part2 = 0.0
        for i in range(self.K):
            Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
            Yg = Y[Ystart:Yend]
            start, end = self.starts[i], self.ends[i]
            xg = x[start:end]
            part2 += np.sum(Yg * xg)
        dual = -(part1+part2)
        return dual, part2


    
    def _projectedGD(self, xk, gradfxk, alphak, params, outter_iter, Y_init=None, outID=None):
        """
            perform projected gradient descent to solve the 
                min 
            termination type:
                params['inexact_type'] 1: ck||s_k||^2
                params['inexact_type'] 2: gamma_2
                params['inexact_type'] 3: O(1/k^3)
        reference: https://sites.math.washington.edu/~burke/crs/408/notes/nlp/gpa.pdf
        The dual variable Y, which is a sparse p-by-K matrix. In stead of storing the whole
        Matrix, we operate on its compressed form.

        """
        uk = xk - alphak * gradfxk
        # ----------------------- pre-screening to select variables to work on ---------
        nonZeroGroupFlag, entrySignFlag, uk = _identifyZeroGroups(uk, alphak, 
                                            self.Lambda_group, 
                                            self.starts, self.ends, self.K)
        print('nonZeroGroupFlag:')
        for i in range(len(nonZeroGroupFlag)):
            print(f" {nonZeroGroupFlag[i]}", end="")
            if (i + 1) % 5 == 0:
                print("")
        print('\nentrySignFlag:')
        for i in range(len(entrySignFlag)):
            print(f" {entrySignFlag[i][0]}", end="")
            if (i + 1) % 5 == 0:
                print("")
        print('work on u:')
        for i in range(len(uk)):
            print(f" {uk[i][0]:3.3e}", end="")
            if (i + 1) % 5 == 0:
                print("")
        # Initialize dual variable Y
        if not Y_init:
            Y = np.zeros((self.r.Yends[-1], 1))
        self.dualDim = 0
        # process Y to make sure its feasibility
        # also Y_{ij} = 0 if x_{i}= 0
        print("===========work on initial Y============")
        for i in range(self.K):
            Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
            if nonZeroGroupFlag[i]:
                self.dualDim += Yend - Ystart
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
        print(' Initial Y is:')
        for i in range(len(Y)):
            print(f" {Y[i][0]:3.3e}", end="")
            if (i + 1) % 5 == 0:
                print("")
        # ----------------- perform the projected gradient descent -------------
        x, gradY = self._xFromY(Y, uk, nonZeroGroupFlag)
        print('\n corresponding x is:')
        for i in range(len(x)):
            print(f" {x[i][0]:3.3e}", end="")
            if (i + 1) % 5 == 0:
                print("")
        fdualY, part2 = self._dual(Y, x, uk)
        iters = 0
        print("Enter the loop for PGD:")
        while True:
            # # ----------------------- check termination -------------------------------------
            # if iters == params['projectedGD']['maxiter']:
            #     flag = 'maxiters'
            #     warnings.warn("inexactness conditon is definitely not satisfied due to the maxiter limit.")
            #     break
            # # check duality gap
            # gap = self.r.func(x) * alphak - part2
            # print(f"gap:{gap}")
            # if gap <= 1e-10:
            #     flag = 'desired '
            #     break
            # ----------------------- one iteration of PGD -------------------------
            print(f"iters:{iters} | PGD stepsize:{params['projectedGD']['stepsize']}")
            Ytrial = self._proj(Y - params['projectedGD']['stepsize'] * gradY, alphak, nonZeroGroupFlag)
            # for i in range(len(Ytrial)):
            #     print(f'Ytrial[{i}]:{Ytrial[i][0]:2.3e}')
            d = Ytrial - Y
            dirder = np.sum(d * gradY)
            #  ------------------------ backtrack line-search ------------------
            stepsize = 1.0
            bak = 0
            print(Y.T)
            while True:
                Ytrial = Y + stepsize * d
                xtrial, _ = self._xFromY(Ytrial, uk, nonZeroGroupFlag)
                # for i in range(len(xtrial)):
                #     print(f'xtrial[{i}]:{xtrial[i][0]:2.3e}')
                fdualYtrial, part2 = self._dual(Ytrial, xtrial, uk)
                lhs =  fdualYtrial - fdualY
                rhs = params['eta'] * stepsize * dirder
                # print(f"lhs: {lhs[0]:3.3e} | rhs: {rhs:3.3e}")
                if lhs <= rhs:
                    break
                if np.abs(lhs) <= 1e-18 and np.abs(rhs) <= 1e-18:
                    break
                if bak > 100:
                    flag = 'lnscfail'
                    warnings.warn("Linesearch failed in projected GD")
                    break
                bak += 1
                stepsize *= params['xi']
            norm_d = utils.l2_norm(d) * stepsize
            print(f"bak:{bak} | stepsize:{stepsize} | norm_d:{norm_d}")
            # ----------------------- check termination -------------------------------------
            if iters == params['projectedGD']['maxiter']:
                flag = 'maxiters'
                warnings.warn("inexactness conditon is definitely not satisfied due to the maxiter limit.")
                break
            # check duality gap
            gap = self.r.func(x) * alphak - part2
            print(f"gap:{gap}")
            if gap <= 1e-10:
                flag = 'desired '
                break
            Y = Ytrial + 0.0
            x = xtrial + 0.0
            print("======")
            iters += 1
        # correct the sign for x
        negx = (entrySignFlag == -1)
        x[negx] = -x[negx]
        return x, Y, flag, iters, gap


            

                
            

        



    

@jit(nopython=True)
def _identifyZeroGroups(uk, alphak, Lambda_group, starts, ends, K):
    # flip the sign of uk, such that uk>=0
    entrySignFlag = np.zeros_like(uk, dtype=np.int8)
    pos_uk, neg_uk = (uk >0).reshape(-1),  (uk<0).reshape(-1)
    entrySignFlag[pos_uk] = 1
    entrySignFlag[neg_uk] = -1
    uk[neg_uk] = -uk[neg_uk]
    # identify some zero-groups
    iterstep = 0
    nonZeroGroupFlag = np.full(K, True)
    # nonZeroBlockFlag = np.full(uk.shape, True)
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
    return nonZeroGroupFlag, entrySignFlag, uk


