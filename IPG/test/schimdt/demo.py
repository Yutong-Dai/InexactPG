'''
File: debug.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-09 16:35
Last Modified: 2021-04-06 01:54
--------------------------------------------
Description:
'''
import sys
from scipy.io import savemat, loadmat
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import src.utils as utils
from src.params import *
from src.regularizer import GL1
from src.lossfunction import LogisticLoss, LeastSquares
from src.schimdt.ProbGL1 import ProbGL1
from src.schimdt.Solver import Solver


test = 'logit'
if test == 'logit':
    datasetName = 'diabetes'
    loss = 'logit'
else:
    # datasetName = 'cpusmall_scale'
    datasetName = 'cadata'
    loss = 'ls'


lam_shrink = 0.1
frac = 0.2
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../db')
if loss == 'logit':
    f = LogisticLoss(X, y, datasetName)
else:
    f = LeastSquares(X, y, datasetName)
p = X.shape[1]
num_of_groups = max(int(p * frac), 2)
group = utils.gen_group(p, num_of_groups)
lammax_path = f'../../../db/lammax-{datasetName}-{frac}.mat'
Lip_path = f'../../../db/Lip-{datasetName}.mat'
if os.path.exists(lammax_path):
    lammax = loadmat(lammax_path)["lammax"][0][0]
    print(f"loading lammax from: {lammax_path}")
else:
    lammax = utils.lam_max(X, y, group, loss)
    savemat(lammax_path, {"lammax": lammax})
    print(f"save lammax to: {lammax_path}")
Lambda = lammax * lam_shrink
r = GL1(Lambda=Lambda, group=group)
prob = ProbGL1(f, r)
params['init_perturb'] = 1e3
params['tol'] = 1e-6
# params['beta'] = 1 / 0.9
params['update_alpha_strategy'] = 'none'
solver = Solver(prob, params)

if os.path.exists(Lip_path):
    L = loadmat(Lip_path)["L"][0][0]
    print(f"loading Lipschitz constant from: {Lip_path}")
else:
    L = utils.estimate_lipschitz(X, loss)
    print(L)
    savemat(Lip_path, {"L": L})
    print(f"save Lipschitz constant to: {Lip_path}")

info = solver.solve(alpha=1 / L)
# info = solver.solve(alpha=1 / L, safeguard_opt='schimdt', safeguard_const=1)
# info = solver.solve(alpha=1 / L, safeguard_opt='const', safeguard_const=1)

# # datasetName = 'a9a'
# datasetName = 'diabetes'
# loss = 'logit'
# lam_shrink = 0.1
# frac = 0.5
# fileType = fileTypeDict[datasetName]
# print("Working on: {}...".format(datasetName))
# X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../db')
# f = LogisticLoss(X, y, datasetName)
# p = X.shape[1]
# num_of_groups = max(int(p * frac), 1)
# group = utils.gen_group(p, num_of_groups)
# lammax_path = f'../../../db/lammax-{datasetName}-{frac}.mat'
# if os.path.exists(lammax_path):
#     lammax = loadmat(lammax_path)["lammax"][0][0]
#     print(f"loading lammax from: {lammax_path}")
# else:
#     lammax = utils.lam_max(X, y, group, loss)
#     savemat(lammax_path, {"lammax": lammax})
#     print(f"save lammax to: {lammax_path}")
# Lambda = lammax * lam_shrink
# r = GL1(Lambda=Lambda, group=group)
# prob = ProbGL1(f, r)
# params['init_perturb'] = 1e3
# params['tol'] = 1e-6
# # params['beta'] = 1 / 0.9
# params['update_alpha_strategy'] = 'model'
# params['method'] = 'schimdt'
# params['delta'] = 1
# # params['update_alpha_strategy'] = 'frac'
# solver = Solver(prob, params)
# info = solver.solve()
