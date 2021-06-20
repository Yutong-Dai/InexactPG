'''
File: debug.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-09 16:35
Last Modified: 2021-06-12 10:23
--------------------------------------------
Description:
'''
import sys
sys.path.append("../..")


import src.utils as utils
from src.params import *
from src.regularizer import natOG
from src.lossfunction import LogisticLoss, LeastSquares
from src.natOG.Problem import ProbNatOG
# from src.natOG.Problem_fulldim import ProbNatOG as ProbNatOGFull
from src.natOG.Solver import Solver


loss = 'logit'
datasetName = 'leu'
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../../GroupFaRSA/db')
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)
else:
    f = LeastSquares(X, y, datasetName)
p = X.shape[1]
grp_size = 10
overlap_ratio = 0.3
generator = utils.GenOverlapGroup(p, grp_size=grp_size, overlap_ratio=overlap_ratio)
starts, ends = generator.get_group()
lammax = utils.lam_max(X, y, starts, ends, loss)
r = natOG(Lambda=lammax * 0.1, dim=p, starts=starts, ends=ends)
# r = natOG(Lambda=0.01, dim=p, starts=starts, ends=ends)
r.createYStartsEnds()
prob = ProbNatOG(f, r)
params['tol'] = 1e-5
params['warm_start'] = True
params['subsolver'] = 'projectedGD'
params['subsolver_verbose'] = False
params['projectedGD']['stepsize'] = 1.0
params['scale_alpha'] = False
params['max_back'] = 50
params['schimdt_const'] = 1e5
params['projectedGD']['maxiter'] = 100
params['update_alpha_strategy'] = 'model'
params['gamma1'] = 1e-12
for i in [1,2,3]:
    params['inexact_type'] = i
    params['projectedGD']['stepsize'] = 1.0
    print(f"inexact type:{params['inexact_type']}")
    solver = Solver(prob, params)
    info = solver.solve(alpha=None, explore=True)
    print(f"F:{info['F']:3.3e} | time:{info['time']:3.3e} | iters:{info['iteration']} | subiters_equiv:{info['subits_equiv']} | nnz:{info['nnz']} | nz:{info['nz']}")


# loss = 'logit'
# datasetName = 'leu'
# fileType = fileTypeDict[datasetName]
# print("Working on: {}...".format(datasetName))
# X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../../GroupFaRSA/db')
# if loss == 'logit':
#     f = LogisticLoss(X, y, datasetName)
# else:
#     f = LeastSquares(X, y, datasetName)
# p = X.shape[1]
# grp_size = min(p // 2, 20)
# overlap_ratio = 0.3
# generator = utils.GenOverlapGroup(p, grp_size=grp_size, overlap_ratio=overlap_ratio)
# starts, ends = generator.get_group()
# lammax = utils.lam_max(X, y, starts, ends, loss)
# r = natOG(Lambda=lammax * 0.1, dim=p, starts=starts, ends=ends)
# # r = natOG(Lambda=0.01, dim=p, starts=starts, ends=ends)
# r.createYStartsEnds()
# prob = ProbNatOG(f, r)
# params['tol'] = 1e-5
# params['warm_start'] = True
# params['subsolver'] = 'projectedGD'
# params['subsolver_verbose'] = False
# params['projectedGD']['stepsize'] = 1.0
# params['scale_alpha'] = False
# params['max_back'] = 50
# params['schimdt_const'] = 1e5
# params['projectedGD']['maxiter'] = 100
# params['update_alpha_strategy'] = 'model'
# params['gamma1'] = 1e-12
# for i in [1,2,3]:
#     params['inexact_type'] = i
#     params['projectedGD']['stepsize'] = 1.0
#     print(f"inexact type:{params['inexact_type']}")
#     solver = Solver(prob, params)
#     info = solver.solve(alpha=None, explore=True)
#     print(f"F:{info['F']:3.3e} | time:{info['time']:3.3e} | iters:{info['iteration']} | subiters_equiv:{info['subits_equiv']} | nnz:{info['nnz']} | nz:{info['nz']}")



