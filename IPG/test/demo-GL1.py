'''
# File: demo.py
# Project: test
# Created Date: 2021-08-30 10:44
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-08-30 12:24
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
from src.regularizer import GL1
from src.solver import IpgSolver
import src.utils as utils
from scipy.io import savemat, loadmat
from src.params import *
import yaml

loss = 'logit'
datasetName = 'diabetes'
frac = 0.3
lam_shrink = 0.8
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
X, y = utils.set_up_xy(datasetName, fileType,
                       dbDir='/Users/ym/Documents/GroupFaRSA/db/')
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)

p = X.shape[1]
num_of_groups = max(int(p * frac), 2)
group = utils.gen_group(p, num_of_groups)
lammax_path = f'/Users/ym/Documents/GroupFaRSA/db/lammax-{datasetName}-{frac}.mat'
Lip_path = f'/Users/ym/Documents/GroupFaRSA/db/Lip-{datasetName}.mat'
if os.path.exists(lammax_path):
    lammax = loadmat(lammax_path)["lammax"][0][0]
    print(f"loading lammax from: {lammax_path}")
else:
    lammax = utils.lam_max(X, y, group, loss)
    savemat(lammax_path, {"lammax": lammax})
    print(f"save lammax to: {lammax_path}")
if os.path.exists(Lip_path):
    L = loadmat(Lip_path)["L"][0][0]
    print(f"loading Lipschitz constant from: {Lip_path}")
else:
    L = utils.estimate_lipschitz(X, loss)
    savemat(Lip_path, {"L": L})
    print(f"save Lipschitz constant to: {Lip_path}")
Lambda = lammax * lam_shrink
r = GL1(penalty=Lambda, groups=group)

with open('../src/config.yaml', "r") as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
solver = IpgSolver(f, r, config)
solver.solve(alpha_init=1.0 / L)
