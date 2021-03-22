'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-07-01 13:49
Last Modified: 2021-03-22 16:47
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


def lam_max(X, y, group, loss='logit'):
    """
    Reference: Yi Yang and Hui Zou. A fast unified algorithm for solving group-lasso penalize learning problems. Page 22.
    """
    # beta_1 = np.log(np.sum(y == 1) / np.sum(y == -1))
    beta_1 = np.zeros((X.shape[1], 1))
    lam_max = -1
    unique_groups, group_frequency = np.unique(group, return_counts=True)
    K = len(unique_groups)
    if loss == 'logit':
        ys = y / (1 + np.exp(y * (X @ beta_1)))
        nabla_L = X.T @ ys / X.shape[0]
    elif loss == 'ls':
        ys = y - X @ beta_1
        nabla_L = (ys.T @ X).T / X.shape[0]
    else:
        raise ValueError("Invalid loss!")
    for i in range(K):
        sub_grp = nabla_L[group == (i + 1)]
        temp = l2_norm(sub_grp) / np.sqrt(group_frequency[i])
        if temp > lam_max:
            lam_max = temp
    return lam_max

# def lam_max(X, y, group):
#     """
#     Reference: Yi Yang and Hui Zou. A fast unified algorithm for solving group-lasso penalize learning problems. Page 22.
#     """
#     # beta_1 = np.log(np.sum(y == 1) / np.sum(y == -1))
#     beta_1 = 0
#     ys = y / (1 + np.exp(beta_1 * y))
#     nabla_L = X.T@ys / X.shape[0]
#     # print(beta_1, norm(ys, 2)**2, nabla_L.T)
#     lam_max = -1
#     unique_groups, group_frequency = np.unique(group, return_counts=True)
#     K = len(unique_groups)
#     for i in range(K):
#         sub_grp = nabla_L[group == (i + 1)]
#         temp = l2_norm(sub_grp) / np.sqrt(group_frequency[i])
#         if temp > lam_max:
#             lam_max = temp
#     return lam_max


def lam_max_jit_prep(X, y, group):
    print("  I am in lam_max_jit_prep", flush=True)
    beta_1 = float(0)
    ys = y / (1 + np.exp(beta_1 * y))
    print('  Compute nabla_L')
    nabla_L = (ys.T @ X).T / X.shape[0]
    if issparse(nabla_L):
        nabla_L = nabla_L.toarray()
        print('  nabla_L to dense...', flush=True)
    print('  Compute unique', flush=True)
    unique_groups, group_frequency = np.unique(group, return_counts=True)
    return unique_groups, group_frequency, nabla_L


@jit(nopython=True, cache=True)
def lam_max_jit(group, unique_groups, group_frequency, nabla_L):
    K = len(unique_groups)
    lam_max = -1
    print('  Going to enter the loop...')
    for i in range(K):
        sub_grp = nabla_L[group == (i + 1)]
        temp = np.sqrt(np.dot(sub_grp.T, sub_grp))[0][0] / np.sqrt(group_frequency[i])
        if temp > lam_max:
            lam_max = temp
        if i % 100000 == 0:
            print("Process")
            print(i)
    return lam_max


def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def get_classification(zeroGroup, nonZeroGroup, zeroProxGroup, nonZeroProxGroup):
    res = {}
    res['Z-Z'] = intersection(zeroGroup, zeroProxGroup)
    res['Z-NZ'] = intersection(zeroGroup, nonZeroProxGroup)
    res['NZ-Z'] = intersection(nonZeroGroup, zeroProxGroup)
    res['NZ-NZ'] = intersection(nonZeroGroup, nonZeroProxGroup)
    return res


def gen_group(p, K):
    dtype = type(K)
    if dtype == int:
        group = K * np.ones(p)
        size = int(np.floor(p / K))
        for i in range(K):
            start_ = i * size
            end_ = start_ + size
            group[start_:end_] = i + 1
    elif dtype == np.ndarray:
        portion = K
        group = np.ones(p)
        chunk_size = p * portion
        start_ = 0
        for i in range(len(portion)):
            end_ = start_ + int(chunk_size[i])
            group[start_:end_] = i + 1
            start_ = int(chunk_size[i]) + start_
    return group


def get_group_structure(X, K, G_i_starts, G_i_ends):
    nz = 0
    for i in range(K):
        start, end = G_i_starts[i], G_i_ends[i]
        X_Gi = X[start:end]
        if (np.sum(np.abs(X_Gi)) == 0):
            nz += 1
    nnz = K - nz
    return nnz


def get_partition(I_cg, I_pg, nonzeroGroup, nonzeroProxGroup, K):
    temp = intersection(nonzeroGroup, nonzeroProxGroup)
    gI_cg = len(temp)
    nI_cg = np.sum(I_cg)
    gI_pg = K - gI_cg
    nI_pg = np.sum(I_pg)
    return gI_cg, nI_cg, gI_pg, nI_pg


class AlgorithmError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
