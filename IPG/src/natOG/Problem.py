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
        self.Lambda_group = self.r.Lambda_group.reshape(-1, 1)
        self.full_group_index = np.arange(self.K)
        self.M = np.sqrt(np.sum(self.Lambda_group ** 2))

    def funcf(self, x):
        return self.f.evaluate_function_value(x)

    def funcr(self, x):
        return self.r.func(x)

    def gradf(self, x):
        # least square and logit loss doesn't require this
        return self.f.gradient()

    def ipg(self, xk, gradfxk, alphak, params, outter_iter, lambda_init=None, outID=None):
        """
            inexactly solve the proximal graident problem
        """
        if params['subsolver'] == 'projectedGD':
            return self._projectedGD(xk, gradfxk, alphak, params, outter_iter, lambda_init, outID)
        else:
            raise ValueError(f"Unknown subsolver:{params['subsolver']}.")
    
    def _duality_gap(self, ):
        primal = None
        dual = None
        gap = primal - dual
        return primal, dual, gap
    
    def _xFromY(self, Y, uk, nonZeroGroupFlag):
        x = uk + 0.0


    
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
        nonZeroGroupFlag, uk = _identifyZeroGroups(uk, alphak, 
                                            self.r.Lambda_group, 
                                            self.r.starts, self.r.ends, self.K)
        # initialize the dual compressed Y
        if not Y_init:
            Y = np.zeros((self.r.Yends[-1],1))
        else:
            Y = Y_init
        
        # sanity operation to make sure that Y is feasible
        for i in range(self.K):
            # this group is non-zero, the assign its a dual variable Yi
            # Yi is a p-by-1 vector, but only [yi]_{gi} is nonzero
            # so in compressed form, Yi is a (self.r.ends[i] - self.r.starts[i])
            # vector.
            Ystart, Yend = self.r.Ystarts[i], self.r.Yends[i]
            if nonZeroGroupFlag[i]:
                start, end = self.r.starts[i], self.r.ends[i]
                ukg =  uk[start:end]
                flag = (ukg == 0.0)
                # Yi
                Y[Ystart:Yend][flag] = 0.0
                Yi_norm = utils.l2_norm(Y[Ystart:Yend])
                if Yi_norm > alphak * self.r.Lambda_group[i]:
                    scale = alphak * self.r.Lambda_group[i] / Yi_norm
                    Y[Ystart:Yend] *= scale
            # this group is zero, hence no dual Yi should be create
            else:
                Y[Ystart:Yend] = 0.0
    


    

@jit(nopython=True)
def _identifyZeroGroups(uk, alphak, Lambda_group, starts, ends, K):
    iterstep = 0
    nonZeroGroupFlag = np.full(uk.shape, True)
    while True:
        iterstep += 1
        if iterstep > K + 1:
            print("The code might have bug.")
        newZeroGroup = 0
        for i in range(K):
            if nonZeroGroupFlag[i]:
                start, end = starts[i], ends[i]
                ukg = uk[start:end]
                ukg_norm = np.sqrt(np.sum(ukg * ukg))
                if ukg_norm <= alphak * Lambda_group[i]:
                    nonZeroGroupFlag[i]= False
                    newZeroGroup += 1
                    uk[start:end] = 0.0
        if newZeroGroup == 0:
            break            
    return nonZeroGroupFlag, uk


