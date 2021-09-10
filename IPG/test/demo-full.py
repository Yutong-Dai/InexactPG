'''
# File: demo-full.py
# Project: test
# Created Date: 2021-09-09 9:27
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-09-10 10:52
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
from scipy.io import savemat, loadmat
from src.lossfunction import LogisticLoss
from src.regularizer import NatOG
from src.solver import IpgSolver
import src.utils as utils
from src.params import *
import yaml
import numpy as np

loss = 'logit'
datasetName = 'a9a'
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
dbDir = '/Users/ym/Documents/GroupFaRSA/db/'
X, y = utils.set_up_xy(datasetName, fileType, dbDir)
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)

p = X.shape[1]
grp_size = 10
overlap_ratio = 0.1
grp_size = min(p // 2, grp_size)
generator = utils.GenOverlapGroup(
    p, grp_size=grp_size, overlap_ratio=overlap_ratio)
groups, starts, ends = generator.get_group()
# print(groups)
# weights = np.ones(len(groups))
# weights[-2] = 100.0

lammax_path = f'{dbDir}/lammax-{datasetName}-{grp_size}-{overlap_ratio}.mat'
if os.path.exists(lammax_path):
    lammax = loadmat(lammax_path)["lammax"][0][0]
    print(f"loading lammax from: {lammax_path}")
else:
    lammax = utils.lam_max(X, y, starts, ends, loss)
    savemat(lammax_path, {"lammax": lammax})
    print(f"save lammax to: {lammax_path}")
lam_shrink = 0.1
r = NatOG(penalty=lammax * lam_shrink, groups=groups, weights=None)


save_ckpt_id = {'date': "09_09_2021", 'loss': loss,
                "lam_shrink": lam_shrink, "grp_size": grp_size, "overlap_ratio": overlap_ratio}
milestone = [1e-3, 1e-4, 1e-5, 1e-6]

alpha_init = 1.0
# print("Exact subprobsolve")
# with open('../src/config.yaml', "r") as stream:
#     config = yaml.load(stream, Loader=yaml.SafeLoader)
# config['mainsolver']['exact_pg_computation'] = True
# solver = IpgSolver(f, r, config)
# info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
#                     save_ckpt_id=save_ckpt_id, milestone=milestone)
# print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# # print(info['x'].T)

# print("Inexact subprobsolve: schimdt")
# with open('../src/config.yaml', "r") as stream:
#     config = yaml.load(stream, Loader=yaml.SafeLoader)
# config['mainsolver']['exact_pg_computation'] = False
# config['mainsolver']['inexact_pg_computation'] = 'schimdt'
# solver = IpgSolver(f, r, config)
# info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
#                     save_ckpt_id=save_ckpt_id, milestone=milestone)
# print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# # print(info['x'].T)

# print("Inexact subprobsolve: lee")
# with open('../src/config.yaml', "r") as stream:
#     config = yaml.load(stream, Loader=yaml.SafeLoader)
# config['mainsolver']['exact_pg_computation'] = False
# config['mainsolver']['inexact_pg_computation'] = 'lee'
# solver = IpgSolver(f, r, config)
# info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
#                     save_ckpt_id=save_ckpt_id, milestone=milestone)
# print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# # print(info['x'].T)

print("Inexact subprobsolve: yd")
with open('../src/config.yaml', "r") as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
config['mainsolver']['exact_pg_computation'] = False
config['mainsolver']['inexact_pg_computation'] = 'yd'
solver = IpgSolver(f, r, config)
info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
                    save_ckpt_id=save_ckpt_id, milestone=milestone)
print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# print(info['x'].T)
