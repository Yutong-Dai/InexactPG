'''
File: Problem.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 01:42
Last Modified: 2021-03-22 21:58
--------------------------------------------
Description:
'''
from abc import ABC, abstractclassmethod
import numpy as np
from numba import jit
import utils
import warnings


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


class ProbGL1(Problem):
    def __init__(self, f, r) -> None:
        super().__init__(f, r)
        self.K = self.r.K
        self.n, self.p = self.f.n, self.f.p
        self.starts = self.r.starts
        self.ends = self.r.ends
        self.Lambda_group = self.r.Lambda_group.reshape(-1, 1)
        self.full_group_index = np.arange(self.K)

    def funcf(self, x):
        return self.f.evaluate_function_value(x)

    def funcr(self, x, method='exact'):
        if method == 'exact':
            return self.r.func_exact(x)
        if method == 'ub':
            return self.r.func_ub(x)
        if method == 'lb':
            return self.r.func_lb(x)

    def gradf(self, x):
        # least square and logit loss doesn't require this
        return self.f.gradient()

    def ipg(self, xk, gradfxk, alphak, params, lambda_init=None):
        """
            perform projected newton to solve the proximal problem
            return the approximation to prox_{alphak,r}(xk-alphak*gradfxk)

            termination type:
                option 1: ck||s_k||^2
                option 2: gamma_2
                option 3: O(1/k^3)
        """
        uk = xk - alphak * gradfxk
        # ----------------------- pre-screening to select variables to be projetced ---------
        to_be_projected_groups = np.full(self.K, False)
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            to_be_projected_groups[i] = utils.l2_norm(xk[start:end]) > self.Lambda_group
        weights_proj_grp = self.Lambda_group[to_be_projected_groups]
        index_proj_grp = self.full_group_index[to_be_projected_groups]
        starts, ends = self.starts[to_be_projected_groups], self.ends[to_be_projected_groups]
        # initialize the dual variable
        if not lambda_init:
            lambda_working = np.zeros((self.K, 1))
        else:
            lambda_working = lambda_init[index_proj_grp]
        lambda_full = np.zeros((self.K, 1))

        iters = 0
        B = len(index_proj_grp)
        if B == 0:
            # in this case, the porjection is equivalent to identify operator
            # z = xk - alphak * gradfxk
            # therefore, the proximal operator is evaluated exactly, which is 0
            x = np.zeros_like(xk)
            return x
        I = np.zeros((self.p, B))
        # set up the indicator functio 1{j\in g}
        for j in range(len(B)):
            I[starts[j]:ends[j], j] = 1
        # ---------------------------- Projected Newton ---------------------------------
        while True:
            iters += 1
            # ----------------------- perform update -------------------------
            # calculate gradient of the dual objective
            denominator = 1 / (1 + I@lambda_working)
            grad = weights_proj_grp**2 - I.T@((uk * denominator)**2)
            epsilon_enlargment = min(0.001, utils.l2_norm(lambda_working - max(0, lambda_working - grad)))
            tmp = 2 * (uk**2) * (denominator**3)

            I_inactive = np.logical_or(grad <= 0.0, lambda_working > epsilon_enlargment)
            I_inactive = np.where(I_inactive == True)
            n_inactive = len(I_inactive)
            B_inactive = np.zeros((n_inactive, n_inactive))

            for j in range(n_inactive):
                _start = starts[I_inactive[j]]
                B_inactive[j,j] = np.sum(tmp[: ])
            # check termination

            # case I: max number of iterations reached
            if iters > params['subprob_maxiter']:
                flag = 'maxiters'
                break
            # case II: desidered termination conditon

    def _duality_gap(self, x, y, xk, gradfxk, alphak):
        gradient_step = xk - alphak * gradfxk
        temp = x - gradient_step
        try:
            primal = np.dot(temp.T, temp)[0][0] / (2 * alphak) + self.r.evaluate_function_value_jit(x)
        except Exception as e:
            print(e)
            primal = 0
            self.overflow = True
        dual_negative = ((alphak / 2) * (np.dot(y.T, y)) + np.dot(gradient_step.T, y))[0][0]
        return primal + dual_negative
