'''
# File: testTree.py
# Project: test
# Created Date: 2021-08-30 6:17
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-08-31 12:38
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


import os
import sys
sys.path.append("../")

from src.lossfunction import LogisticLoss
from src.regularizer import TreeOG
from src.solver import IpgSolver
import src.utils as utils
from scipy.io import savemat, loadmat
from src.params import *
import yaml
import numpy as np
from scipy.sparse import csc_matrix

loss = 'logit'
datasetName = 'diabetes'
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
X, y = utils.set_up_xy(datasetName, fileType,
                       dbDir='/Users/ym/Documents/GroupFaRSA/db/')
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)

Lip_path = f'/Users/ym/Documents/GroupFaRSA/db/Lip-{datasetName}.mat'
if os.path.exists(Lip_path):
    L = loadmat(Lip_path)["L"][0][0]
    print(f"loading Lipschitz constant from: {Lip_path}")
else:
    L = utils.estimate_lipschitz(X, loss)
    savemat(Lip_path, {"L": L})
    print(f"save Lipschitz constant to: {Lip_path}")

g0 = [0, 1, 2, 3, 4, 5, 6, 7]
g1 = [2, 3, 4]
g2 = [5, 6, 7]
groups = [g0, g1, g2]
weights = np.array([1.0, 20.0, 1.0])
penalty = 0.1

# pointer to the first variable of each group
own_variables = np.array([0, 2, 5], dtype=np.int32)
# number of "root" variables in each group
N_own_variables = np.array([2, 3, 3], dtype=np.int32)
# (variables that are in a group, but not in its descendants).
# for instance root(g1)={0,1}, root(g2)={2 3 4}, root(g3)={5 6 7 8 9}
# weights for each group, they should be non-zero to use fenchel duality
eta_g = weights * penalty
_groups_bool = np.asfortranarray([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 0, 0]], dtype=np.bool)
# first group should always be the root of the tree
# non-zero entriees mean inclusion relation ship, here g2 is a children of g1,
# g3 is a children of g1
_groups_bool = csc_matrix(_groups_bool, dtype=np.bool)
tree = {'eta_g': eta_g, 'groups': _groups_bool, 'own_variables': own_variables,
        'N_own_variables': N_own_variables}

r = TreeOG(penalty=0.1, tree=tree, groups=groups, weights=weights)

with open('../src/config.yaml', "r") as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
config['subsolver']['compute_exactpg'] = False
solver = IpgSolver(f, r, config)
solver.solve(alpha_init=1.0 / L)
print(solver.solution.T)
