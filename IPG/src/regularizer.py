'''
File: regularizer.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-04-18 10:18
Last Modified: 2021-04-18 10:36
--------------------------------------------
Description:
'''
import numpy as np
from numba import jit
import cvxpy as cp


class OGL1:
    def __init__(self, Lambda, dim, starts, ends):
        """
        Lambda: scalar > 0
        starts: a list of numbers speficy the starting index of each group
          ends: a list of numbers speficy the end index of each group

        For example, a overlapping group configuration, the number stands for
        the index of the variable.
        {g0=[0,1,2,3,4],g1=[3,4,5,6,7], g2=[5,6,7,8,9]}
        stars = [0, 3, 5]
        ends  = [4, 7, 9]
        """
        self.p = dim
        self.K = len(starts)
        # a np.array that stores the number of group that each coordinate belongs to
        self.freq = np.zeros((self.p, 1))
        self.group_size = np.zeros(self.K)
        self.groups = {}
        for i in range(self.K):
            self.freq[starts[i]:ends[i] + 1] += 1
            self.group_size[i] = ends[i] - starts[i] + 1
            self.groups[i] = np.arange(starts[i], ends[i] + 1)
        self.Lambda_group = Lambda * np.sqrt(self.group_size)
        self.starts = np.array(starts)
        # since python `start:end` will include `start` and exclude `end`,
        # we add 1 to the `end` so the G_i-th block of X is indexed by X[start:end]
        self.ends = np.array(ends) + 1

    def __str__(self):
        return("Overlapping Group L1")

    def func_exact(self, X):
        w = self.Lambda_group
        y = cp.Variable(self.p)
        soc_constraints = []
        for i in range(len(self.starts)):
            # note here self.ends is shifted by 1
            soc_constraints.append(cp.norm(y[self.starts[i]:self.ends[i]], 2) <= w[i])
        prob = cp.Problem(cp.Maximize(X.T @ y), soc_constraints)
        prob.solve(solver=cp.SCS)
        return prob.value

    def func_ub(self, X):
        return _fub1_jit(X, self.K, self.starts, self.ends, self.Lambda_group, self.freq)
    # def func_ub(self, X, approx=1):
    #     if approx == 1:
    #         return _fub1_jit(X, self.K, self.starts, self.ends, self.Lambda_group, self.freq)
    #     else:
    #         return _fub2_jit(X, self.K, self.starts, self.ends, self.Lambda_group)

    def func_lb(self, X):
        return _flb_jit(X, self.Lambda_group)

    def dual(self, y):
        return _dual_jit(y, self.K, self.starts, self.ends, self.Lambda_group)


@jit(nopython=True)
def _fub1_jit(X, K, starts, ends, Lambda_group, freq):
    ub = 0.0
    for i in range(K):
        start, end = starts[i], ends[i]
        # decompose X_g into V_g
        Vg = X[start:end] / freq[start:end]
        ub += Lambda_group[i] * np.sqrt(np.sum(Vg * Vg))
    return ub


# @jit(nopython=True)
# def _fub2_jit(X, K, starts, ends, Lambda_group):
#     ub = 0.0
#     for i in range(K):
#         start, end = starts[i], ends[i]
#         # decompose X_g into V_g
#         Vg = X[start:end]
#         ub += Lambda_group[i] * np.sqrt(np.sum(Vg * Vg))
#         X[start:end] = 0.0
#     return ub


@jit(nopython=True)
def _flb_jit(X, Lambda_group):
    y = X / np.sqrt(np.sum(X * X))
    y = min(Lambda_group) * y
    lb = np.dot(X.T, y)[0][0]
    return lb


@jit(nopython=True)
def _dual_jit(y, K, starts, ends, Lambda_group):
    max_group_norm = 0.0
    for i in range(K):
        start, end = starts[i], ends[i]
        yg = y[start:end]
        temp = (np.sqrt(np.dot(yg * yg))[0][0]) / Lambda_group[i]
        max_group_norm = max(max_group_norm, temp)
    return max_group_norm
