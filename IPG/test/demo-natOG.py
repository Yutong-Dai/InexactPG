'''
# File: demo.py
# Project: test
# Created Date: 2021-08-30 10:44
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-08-30 6:39
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
from src.regularizer import NatOG
from src.solver import IpgSolver
import src.utils as utils
from scipy.io import savemat, loadmat
from src.params import *
import yaml
import numpy as np

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
r = NatOG(penalty=0.1, groups=groups, weights=weights)

with open('../src/config.yaml', "r") as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
config['subsolver']['compute_exactpg'] = False
solver = IpgSolver(f, r, config)
solver.solve(alpha_init=1.0 / L)
print(solver.solution.T)
