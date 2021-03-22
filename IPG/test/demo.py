'''
File: debug.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-09 16:35
Last Modified: 2021-02-28 18:17
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
from src.baseKappa.solve import solve

loss = 'logit'
hard = False
if not hard:
    if loss == 'ls':
        # datasetName = 'bodyfat_scale'
        datasetName = 'YearPredictionMSD.t'
        lam_shrink = 0.01
        frac = 0.25
        fileType = fileTypeDict[datasetName]
        print("Working on: {}...".format(datasetName))
        X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../db_raw')
        f = LeastSquares(X, y, datasetName)
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
        info = solve(f, r, X_initial=None, proxStepsize=None, method='gradient',
                     update_proxStepsize='single', params=params, print_group=False, print_second_level=False,
                     kappa_1=1e-1, kappa_2=1e-2, print_time=True)
    if loss == 'logit':
        datasetName = 'a9a'
        lam_shrink = 0.01
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
        # Lambda = utils.lam_max(X, y, group) * lam_shrink
        r = GL1(Lambda=Lambda, group=group)
        info = solve(f, r, X_initial=None, proxStepsize=None, method='gradient',
                     update_proxStepsize='single', params=params, print_group=False, print_second_level=False,
                     kappa_1=1e-1, kappa_2=1e-2, print_time=True)
        # print(info['X'])

else:
    datasetName = 'gisette'
    lam_shrink = 0.01
    frac = 0.25
    fileType = fileTypeDict[datasetName]
    print("Working on: {}...".format(datasetName))
    X, y = utils.set_up_xy(datasetName, fileType, dbDir='../../../db')
    f = LogisticLoss(X, y, datasetName)
    p = X.shape[1]
    num_of_groups = max(int(p * frac), 1)
    group = utils.gen_group(p, num_of_groups)
    Lambda = utils.lam_max(X, y, group) * lam_shrink
    r = GL1(Lambda=Lambda, group=group)
    info = solve(f, r, X_initial=None, proxStepsize=None, method='gradient',
                 update_proxStepsize='single', params=params, print_group=False, print_second_level=False,
                 kappa_1=1, kappa_2=1, print_time=True)

print("Fval: {:3.3e}".format(info['F']))
print(f"nnz: {info['nnz']}")
print("Time: {:6.3f}".format(info['time']))
