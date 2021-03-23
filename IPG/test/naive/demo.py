'''
File: debug.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-09 16:35
Last Modified: 2021-03-23 16:52
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
from src.lossfunction import LogisticLoss
from src.naive.ProbGL1 import ProbGL1
from src.naive.Solver import Solver


# datasetName = 'a9a'
datasetName = 'diabetes'
loss = 'logit'
lam_shrink = 0.1
frac = 0.5
fileType = fileTypeDict[datasetName]
print("Working on: {}...".format(datasetName))
X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../db')
f = LogisticLoss(X, y, datasetName)
p = X.shape[1]
num_of_groups = max(int(p * frac), 1)
group = utils.gen_group(p, num_of_groups)
lammax_path = f'../../../db/lammax-{datasetName}-{frac}.mat'
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
params['init_perturb'] = 0.1
params['tol'] = 1e-6
# params['beta'] = 1 / 0.9
params['update_alpha_strategy'] = 'model'
# params['update_alpha_strategy'] = 'frac'
solver = Solver(prob, params)
info = solver.solve()
