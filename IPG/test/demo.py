'''
File: debug.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-09 16:35
Last Modified: 2021-05-27 22:42
--------------------------------------------
Description:
'''
import sys
from scipy.io import savemat, loadmat
import os
sys.path.append("../")


import src.utils as utils
from src.params import *
from src.regularizer import OGL1
from src.lossfunction import LogisticLoss, LeastSquares
from src.Problem import ProbOGL1
from src.Solver import Solver
import numpy as np

test = 'logit'
# test = 'ls'
if test == 'logit':
    datasetName = "diabetes"
    # datasetName = 'a9a'
    # datasetName = 'w8a'
    loss = 'logit'
else:
    # datasetName = 'cpusmall_scale'
    datasetName = 'cadata'
    loss = 'ls'


lam_shrink = 0.1
fileType = fileTypeDict[datasetName]

print("Working on: {}...".format(datasetName))
X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../../GroupFaRSA/db')
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)
else:
    f = LeastSquares(X, y, datasetName)
p = X.shape[1]
num_grp, grp_size = 3, 5
# lammax_path = f'../../../db/lammax-{datasetName}-{frac}.mat'
# if os.path.exists(lammax_path):
#     lammax = loadmat(lammax_path)["lammax"][0][0]
#     print(f"loading lammax from: {lammax_path}")
# else:
#     lammax = utils.lam_max(X, y, group, loss)
#     savemat(lammax_path, {"lammax": lammax})
#     print(f"save lammax to: {lammax_path}")
Lip_path = f'../../../db/Lip-{datasetName}.mat'
if os.path.exists(Lip_path):
    L = loadmat(Lip_path)["L"][0][0]
    print(f"loading Lipschitz constant from: {Lip_path}")
else:
    L = utils.estimate_lipschitz(X, loss)
    savemat(Lip_path, {"L": L})
    print(f"save Lipschitz constant to: {Lip_path}")

# Lambda = lammax * lam_shrink
generator = utils.GenOverlapGroup(p, num_grp, grp_size)
starts, ends = generator.get_group()
r = OGL1(Lambda=10, dim=p, starts=starts, ends=ends)
prob = ProbOGL1(f, r)

solver = Solver(prob, params)
info = solver.solve(alpha=1 / L, explore=True)
