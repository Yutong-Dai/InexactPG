'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-07-01 13:49
Last Modified: 2021-04-05 20:51
--------------------------------------------
Description:
'''
import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
from scipy.sparse import issparse
from numba import jit


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
    def __init__(self, dim, num_grp, grp_size):
        self.dim = dim
        self.num_grp = num_grp
        self.grp_size = grp_size

    def get_group(self):
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
        return starts, ends


def lam_max(X, y, group, loss='logit'):
    """
    """
    pass


class AlgorithmError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
