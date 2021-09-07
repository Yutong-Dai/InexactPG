'''
File: lossfunction.py
Author: Yutong Dai 
File Created: 2020-05-11 18:03
Last Modified: 2021-01-27 01:33
--------------------------------------------
Description:
'''
import numpy as np


class SimpleQudratic:
    def __init__(self, Q, b, c):
        self.Q = Q
        self.b = b
        self.c = c
        self.datasetName = "randSQ"

    def func(self, x):
        return 0.5 * (x.T @ self.Q @ x) + self.b.T @ x + self.c

    def grad(self, x):
        return self.Q @ x + self.b


class LogisticLoss:
    def __init__(self, X, y, datasetName=None):
        """
        X is the data matrix
        y is the label
        """
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.expterm = None
        self.sigmoid = None
        self.datasetName = datasetName

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}Logistic\n".format(
            '', self.n, self.p, '')
        return info

    def func(self, weight):
        """
        function value of logistic loss function evaluate at the given point weight.
        f(weight) = frac{1}{n} sum_{i=1}^n log(1+exp(y_i * weight^T * x_i))
        """
        self.expterm = np.exp(-(self.y) * (self.X @ weight))
        f = np.sum(np.log(1 + self.expterm)) / self.n
        return f

    def grad(self, weight):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        self.sigmoid = 1 - (1 / (1 + self.expterm))
        gradient = -((self.sigmoid * self.y).T @ self.X) / self.n
        return gradient.T

    def _prepare_hv_data(self, col_idx):
        self.X_subset = self.X[:, col_idx]
        self.sigmoid_prod = (self.sigmoid * (1 - self.sigmoid))
        self.sigmoid_prod = np.maximum(1e-8, self.sigmoid_prod)

    def hessian_vector_product_fast(self, v):
        """
        call _prepare_hv_data before hessian_vector_product_fast to have sigmoid_prod, X_subset quantities being stored
        """
        A = (self.X_subset @ v)
        B = A * self.sigmoid_prod
        hv = (B.T @ self.X_subset).T / self.n
        return hv


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
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}LeastSquares\n".format(
            '', self.n, self.p, '')
        return info

    def func(self, weight):
        """
        function value of logistic loss function evaluate at the given point weight.
        f(weight) = frac{1}{n} ||X*weight-y||^2
        """
        self.matvec = self.X @ weight - self.y
        f = 0.5 * np.sum(self.matvec * self.matvec) / self.n
        return f

    def grad(self, weight):
        """
        need to be called after `evaluate_function_value` to get correct `matvec`
        """
        gradient = self.matvec.T @ self.X / self.n
        return gradient.T

    def _prepare_hv_data(self, col_idx):
        self.X_subset = self.X[:, col_idx]

    def hessian_vector_product_fast(self, v):
        temp = (self.X_subset @ v)
        hv = (temp.T @ self.X_subset).T / self.n
        return hv + 1e-8 * v
