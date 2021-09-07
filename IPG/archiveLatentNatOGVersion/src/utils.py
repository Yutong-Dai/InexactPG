'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-07-01 13:49
Last Modified: 2021-06-12 00:11
--------------------------------------------
Description:
'''
import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
from scipy.sparse import issparse
from numba import jit
import warnings


def l2_norm(x):
    return np.sqrt(np.dot(x.T, x))[0][0]


def linf_norm(x):
    return np.max(np.abs(x))


def set_up_xy(datasetName, fileType='txt', dbDir='../db', to_dense=False):
    filepath = dbDir + "/{}.{}".format(datasetName, fileType)
    if fileType != 'mat':
        data = load_svmlight_file(filepath)
        X, y = data[0], data[1].reshape(-1, 1)
        # if datasetName in ['gisette']:
        #     to_dense = True
        if to_dense:
            print("  Begin converting {}...".format(datasetName))
            X = X.toarray()
            print("  Finish converting!")
        return X, y
    else:
        data_dict = loadmat(filepath)
        try:
            return data_dict['A'], data_dict['b']
        except KeyError:
            print("Invalid matlab data file path... I cannot find X and y.")


class GenOverlapGroup:
    def __init__(self, dim, num_grp=None, grp_size=None, overlap_ratio=None):
        self.dim = dim
        self.num_grp = num_grp
        self.grp_size = grp_size
        self.overlap_ratio = overlap_ratio

    def get_group(self):
        if self.num_grp is not None and self.grp_size is not None:
            if self.grp_size * self.num_grp <= self.dim:
                raise ValueError("grp_size is too small to have overlapping.")
            if self.grp_size >= self.dim:
                raise ValueError("grp_size is too large that each group has all variables.")
            exceed = self.num_grp * self.grp_size - self.dim
            overlap_per_group = int(exceed / (self.num_grp - 1))
            starts = [0] * self.num_grp
            ends = [0] * self.num_grp
            for i in range(self.num_grp):
                if i == 0:
                    start = 0
                    end = start + self.grp_size - 1
                else:
                    start = end - overlap_per_group + 1
                    end = min(start + self.grp_size - 1, self.dim - 1)
                    if start == starts[i - 1] and end == ends[i - 1]:
                        self.num_grp = i
                        print(f"The actual number of group is {self.num_grp}")
                        break
                starts[i] = start
                ends[i] = end
            return starts[:i + 1], ends[:i + 1]
        elif self.grp_size is not None and self.overlap_ratio is not None:
            if self.grp_size >= self.dim:
                raise ValueError("grp_size is too large that each group has all variables.")
            overlap = int(self.grp_size * self.overlap_ratio)
            if overlap < 1:
                msg = "current config of grp_size and overlap_ratio cannot produce overlapping groups.\n"
                msg += "overlap_ratio is adjusted to have at least one overlap."
                warnings.warn(msg)
                overlap = 1
            starts = []
            ends = []
            self.num_grp = 0
            start = 0
            end = self.grp_size - 1
            while True:
                starts.append(start)
                ends.append(end)
                self.num_grp += 1
                # update
                start = end - (overlap - 1)
                end = min(start + self.grp_size - 1, self.dim - 1)
                if end == ends[-1]:
                    break
            return starts, ends
        else:
            raise ValueError("check your inputs!")


def lam_max(X, y, starts, ends, loss='logit'):
    """
    Reference: Yi Yang and Hui Zou. A fast unified algorithm for solving group-lasso penalize learning problems. Page 22.
    """
    beta_1 = np.zeros((X.shape[1], 1))
    lam_max = -1
    K = len(starts)
    if loss == 'logit':
        ys = y / (1 + np.exp(y * (X @ beta_1)))
        nabla_L = X.T @ ys / X.shape[0]
    elif loss == 'ls':
        ys = y - X @ beta_1
        nabla_L = (ys.T @ X).T / X.shape[0]
    else:
        raise ValueError("Invalid loss!")
    for i in range(K):
        start, end = starts[i], ends[i] + 1
        sub_grp = nabla_L[start:end]
        temp = l2_norm(sub_grp) / np.sqrt(end - start)
        if temp > lam_max:
            lam_max = temp
    return lam_max


def estimate_lipschitz(A, loss='logit'):
    m, n = A.shape
    if loss == 'ls':
        hess = A.T @ A / m
    elif loss == 'logit':
        # acyually this is an upper bound on hess
        hess = A.T @ A / (4 * m)
    hess = hess.toarray()
    L = np.max(np.linalg.eigvalsh(hess))
    return L


class AlgorithmError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
