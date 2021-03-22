'''
File: regularizer.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-05-11 21:18
Last Modified: 2021-03-22 14:41
--------------------------------------------
Description:
'''
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix
try:
    import utils
except (ImportError, ModuleNotFoundError):
    import sys
    import os
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    import src.utils as utils


@jit(nopython=True)
def _f(X, K, starts, ends, Lambda_group):
    fval = 0.0
    for i in range(K):
        start, end = starts[i], ends[i]
        XG_i = X[start:end]
        fval += Lambda_group[i] * np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
    return fval


@jit(nopython=True)
def _dual_jit(y, K, starts, ends, Lambda_group):
    """
    compute the dual of r(x), which is r(y): max ||y_g||/lambda_g
    reference: https://jmlr.org/papers/volume18/16-577/16-577.pdf section 5.2
    """
    max_group_norm = 0.0
    for i in range(K):
        start, end = starts[i], ends[i]
        yG_i = y[start:end]
        temp_i = (np.sqrt(np.dot(yG_i.T, yG_i))[0][0]) / Lambda_group[i]
        max_group_norm = max(max_group_norm, temp_i)
    return max_group_norm


class GL1:
    def __init__(self, Lambda, group):
        """
        !!Warning: need `group` be ordered in a consecutive manner, i.e., 
        group: array([1., 1., 1., 2., 2., 2., 3., 3., 3., 3.])
        Then:
        unique_groups: array([1., 2., 3.])
        group_frequency: array([3, 3, 4]))
        """
        self.group = group
        self.Lambda = Lambda
        self.unique_groups, self.group_frequency = np.unique(self.group, return_counts=True)
        self.Lambda_group = self.Lambda * np.sqrt(self.group_frequency)
        self.K = len(self.unique_groups)
        self.group_size = -1 * np.ones(self.K)
        p = group.shape[0]
        full_index = np.arange(p)
        starts = []
        ends = []
        data = np.zeros(p)
        row_idx = np.zeros(p)
        col_idx = np.zeros(p)
        for i in range(self.K):
            G_i = full_index[np.where(self.group == self.unique_groups[i])]
            # record the `start` and `end` indices of the group G_i to avoid fancy indexing innumpy
            # in the example above, the start index and end index for G_1 is 0 and 2 respectively
            # since python `start:end` will include `start` and exclude `end`, so we will add 1 to the `end`
            # so the G_i-th block of X is indexed by X[start:end]
            start, end = min(G_i), max(G_i) + 1
            starts.append(start)
            ends.append(end)
            self.group_size[i] = end - start
            data[start:end] = self.Lambda_group[i] ** 2
            row_idx[start:end] = i
            col_idx[start:end] = np.arange(start, end)
        # wrap as np.array for jit compile purpose
        self.starts = np.array(starts)
        self.ends = np.array(ends)
        self.mat = csr_matrix((data, (row_idx, col_idx)), shape=(self.K, p))

    def __str__(self):
        return("Group L1")

    def eval_f_vec(self, X):
        return np.sum(np.sqrt(self.mat @ (X ** 2)))

    def evaluate_function_value_jit(self, X):
        return _f(X, self.K, self.starts, self.ends, self.Lambda_group)
    # gradient is calculated on the fly, no need to define a method here.

    def dual(self, y):
        return _dual_jit(y, self.K, self.starts, self.ends, self.Lambda_group)

    def _prepare_hv_data(self, X, subgroup_index):
        self.hv_data = {}
        start = 0
        for i in subgroup_index:
            start_x, end_x = self.starts[i], self.ends[i]
            XG_i = X[start_x:end_x]
            XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            end = start + end_x - start_x
            self.hv_data[i] = {}
            self.hv_data[i]['XG_i'] = XG_i
            self.hv_data[i]['XG_i_norm'] = XG_i_norm
            self.hv_data[i]['start'] = start
            self.hv_data[i]['end'] = end
            self.hv_data[i]['XG_i_norm_cubic'] = XG_i_norm**3
            start = end

    def hessian_vector_product_fast(self, v, subgroup_index):
        hv = np.empty_like(v)
        for i in subgroup_index:
            start = self.hv_data[i]['start']
            end = self.hv_data[i]['end']
            vi = v[start:end]
            temp = np.matmul(self.hv_data[i]['XG_i'].T, vi)
            hv[start:end] = self.Lambda_group[i] * (1 / self.hv_data[i]['XG_i_norm'] * vi -
                                                    (temp / self.hv_data[i]['XG_i_norm_cubic']) *
                                                    self.hv_data[i]['XG_i'])
        return hv

    # following are provided for proof of concepts purpose
    # not meant to be efficient
    def evaluate_function_value(self, X):
        fval = 0
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            XG_i = X[start:end]
            fval += self.Lambda_group[i] * utils.l2_norm(XG_i)
        return fval

    def gradient(self, X, epsilon=1e-10):
        gradient = np.zeros_like(X)
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            XG_i = X[start:end]
            # XG_i_norm = norm(XG_i, 2)
            XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            if XG_i_norm > epsilon:
                gradient[start:end] = self.Lambda_group[i] * XG_i / XG_i_norm
        return gradient

    def hessian_vector_product(self, X, v, subgroup_index, epsilon=1e-8):
        hv = np.empty_like(v)
        start = 0
        for i in range(subgroup_index.shape[0]):
            start, end = self.starts[i], self.ends[i]
            XG_i = X[start:end]
            # XG_i_norm = utils.l2_norm(XG_i)
            XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            vi = v[start:end]
            temp = np.matmul(XG_i.T, vi)
            hv[start:end] = self.Lambda_group[subgroup_index[i]] * (1 / XG_i_norm * vi - (temp / XG_i_norm**3) * XG_i) + vi * 1e-8
            start = end
        return hv

    def hessian(self, X):
        hessian = np.zeros([X.shape[0], X.shape[0]])
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            XG_i = X[start:end]
            XG_i_norm = utils.l2_norm(XG_i)
            if XG_i_norm != 0:
                temp_i = (np.eye(np.sum(len(XG_i))) / XG_i_norm) - (np.matmul(XG_i, XG_i.T) / np.power(XG_i_norm, 3))
                hessian[start:end, :][:, start:end] = self.Lambda_group[i] * temp_i
        hessian += np.diag(1e-8 * np.ones(X.shape[0]))
        return hessian
