'''
File: lossfunction.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-05-11 18:03
Last Modified: 2021-01-27 01:33
--------------------------------------------
Description:
'''
import numpy as np
from scipy import sparse
from numpy.linalg import eigvalsh
from numba import jit


# @jit(nopython=True)
def _gamma(X, starts, ends, K):
    gammas = np.zeros(K)
    for i in range(K):
        start, end = starts[i], ends[i]
        Xsub = X[:, start:end]
        gammas[i] = max(eigvalsh(Xsub.T@Xsub))
    return gammas


class LogisticLoss:
    def __init__(self, X, y, datasetName=None):
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.expterm = None
        self.sigmoid = None
        self.datasetName = datasetName

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}Logistic\n".format('', self.n, self.p, '')
        return info

    def evaluate_function_value(self, weight, bias=0):
        """
        function value of logistic loss function evaluate at the given point (weight, bias).
        f(weight,bias) = frac{1}{n} sum_{i=1}^n log(1+exp(y_i * weight^T * x_i))
        """
        self.expterm = np.exp(-(self.y) * (self.X@weight + bias))
        f = np.sum(np.log(1 + self.expterm)) / self.n
        return f

    def evaluate_function_value_at_multiple_points(self, weightMat):
        """
        currently not support the bias vector. To do in the future.
        """
        exptermMat = np.exp(-(self.y) * (self.X@weightMat))
        # fvec = [f1, f2, f3, ....] is a row vector
        fvec = np.sum(np.log(1 + exptermMat), axis=0) / self.n
        return fvec

    def gradient(self):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        self.sigmoid = 1 - (1 / (1 + self.expterm))
        gradient = -((self.sigmoid * self.y).T@self.X) / self.n
        return gradient.T

    def _prepare_hv_data(self, subgroup_index):
        self.X_subset = self.X[:, subgroup_index]
        self.sigmoid_prod = (self.sigmoid * (1 - self.sigmoid))
        self.sigmoid_prod = np.maximum(1e-8, self.sigmoid_prod)

    def hessian_vector_product_fast(self, v):
        A = (self.X_subset@v)
        B = A * self.sigmoid_prod
        hv = (B.T@self.X_subset).T / self.n
        return hv

    def _prepare_hv_approx_data(self, subgroup_index):
        self.X_subset = self.X[:, subgroup_index]

    def hv_approx(self, v):
        A = self.X_subset@v
        # avoid sparseMat * sparseMat * denseVec
        hv = (A.T@self.X_subset).T / (4 * self.n)
        return hv

    def _prepare_D(self):
        self.sigmoid_prod = (self.sigmoid * (1 - self.sigmoid))
        self.sigmoid_prod = np.maximum(1e-8, self.sigmoid_prod)

    def _prepare_data_mat(self, start, end):
        self.X_subset = self.X[:, start:end]
        # using numpy broadcasting
        #  [1]   [1, 2, 3]      [1, 2, 3]
        #      *             =
        #  [2]   [1, 2, 3]      [2, 4, 6]
        # self.DXsub = self.sigmoid_prod * self.X_subset

    def _hv_Gi(self, v):
        B = (self.X_subset@v) * self.sigmoid_prod
        hv = (B.T@self.X_subset).T / self.n
        return hv

    def get_group_hess_max_eign(self, starts, ends, K):
        """
            Compute the groupwise largest eigenvalue of the  matrix, which is an upper bound on 
            the groupwise Hessian
        """
        gammas = _gamma(self.X, starts, ends, K)
        gammas = gammas * (1 + 1e-6) * 0.25 / self.n
        return gammas

    # following are provided for proof of concepts purpose
    # not meant to be efficient

    def hessian_vector_product(self, v, subgroup_index):
        X_subset = self.X[:, subgroup_index]
        A = (X_subset@v)
        B = A * (self.sigmoid * (1 - self.sigmoid))
        hv = (B.T@X_subset).T / self.n
        return hv

    def hessian_approx(self):
        h = self.X.T@self.X / (4 * self.n)
        return h

    def hessian(self):
        sigmoid_prod = self.sigmoid * (1 - self.sigmoid)
        D = np.maximum(1e-8, sigmoid_prod)
        if sparse.issparse(self.X):
            D = sparse.diags(D.reshape(-1,))
            h = (self.X.T@D)@self.X / self.n
            h = h.toarray()
        else:
            D = np.diag(D.reshape(-1,))
            h = (self.X.T@D)@self.X / self.n
        # h += np.float32(1e-8) * np.eye(h.shape[0], dtype=np.float32)
        return h


class LeastSquares:
    def __init__(self, X, y, datasetName=None):
        """
        docstring
        """
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.datasetName = datasetName

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}LeastSquares\n".format('', self.n, self.p, '')
        return info

    def evaluate_function_value(self, weight):
        """

        """
        self.matvec = self.X@weight - self.y
        f = 0.5 * np.sum(self.matvec * self.matvec) / self.n
        return f

    def gradient(self):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        gradient = self.matvec.T @ self.X / self.n
        return gradient.T

    def _prepare_hv_data(self, subgroup_index):
        self.X_subset = self.X[:, subgroup_index]

    def hessian_vector_product_fast(self, v):
        temp = (self.X_subset@v)
        hv = (temp.T@self.X_subset).T / self.n
        return hv + 1e-8 * v

    def hessian(self):
        return self.X.T@self.X / self.n
