'''
# File: demo-full.py
# Project: test
# Created Date: 2021-09-09 9:27
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-09-10 11:34
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
from src.lossfunction import LogisticLoss, LeastSquares
from src.regularizer import NatOG
from src.solver import IpgSolver
from src.solverlee import IpgSolverLee
import src.utils as utils
from src.params import *
import yaml
import platform
import argparse
import numpy as np


loss = 'logit'
datasetName = 'w8a'
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
# dbDir = '/Users/ym/Documents/GroupFaRSA/db'
# dbDir = '../../../GroupFaRSA/db_big'
dbDir = '../../../GroupFaRSA/db'
X, y = utils.set_up_xy(datasetName, fileType, dbDir)
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)

p = X.shape[1]
grp_size = 100
overlap_ratio = 0.1
grp_size = min(p // 2, grp_size)
generator = utils.GenOverlapGroup(
    p, grp_size=grp_size, overlap_ratio=overlap_ratio)
groups, starts, ends = generator.get_group()


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


save_ckpt_id = {'date': "09_01_2021", 'loss': loss,
                "lam_shrink": lam_shrink, "grp_size": grp_size, "overlap_ratio": overlap_ratio}
milestone = [1e-3, 1e-4, 1e-5]

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
# config['inexactpg']['schimdt']['c'] = 1e4
# solver = IpgSolver(f, r, config)
# info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
#                     save_ckpt_id=save_ckpt_id, milestone=milestone)
# print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# print(info['x'].T)

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

# print("Inexact subprobsolve: yd")
# with open('../src/config.yaml', "r") as stream:
#     config = yaml.load(stream, Loader=yaml.SafeLoader)
# config['mainsolver']['exact_pg_computation'] = False
# config['mainsolver']['inexact_pg_computation'] = 'yd'
# config['inexactpg']['yd']['gamma'] = 0.1
# # config['subsolver']['iteration_limits'] = 2
# solver = IpgSolver(f, r, config)
# info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
#                     save_ckpt_id=save_ckpt_id, milestone=milestone)
# print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# print(info['x'].T)


# ========

print("Mainsover: Lee |  Inexact subprobsolve: lee")
with open('../src/config.yaml', "r") as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
config['mainsolver']['exact_pg_computation'] = False
config['mainsolver']['inexact_pg_computation'] = 'lee'
solver = IpgSolverLee(f, r, config)
info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
                    save_ckpt_id=save_ckpt_id, milestone=milestone)
print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")


print("Mainsover: Mine | Inexact subprobsolve: lee")
with open('../src/config.yaml', "r") as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
config['mainsolver']['exact_pg_computation'] = False
config['mainsolver']['inexact_pg_computation'] = 'lee'
solver = IpgSolver(f, r, config)
info = solver.solve(alpha_init=alpha_init, save_ckpt=True,
                    save_ckpt_id=save_ckpt_id, milestone=milestone)
print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
# print(info['x'].T)
