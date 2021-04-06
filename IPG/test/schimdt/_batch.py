'''
File: batch.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-21 00:46
Last Modified: 2021-04-06 02:19
--------------------------------------------
Description:
'''

import sys
import os
from scipy.io import savemat, loadmat
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np

import src.utils as utils
from src.params import fileTypeDict
from src.regularizer import GL1
from src.lossfunction import LogisticLoss, LeastSquares
from src.schimdt.ProbGL1 import ProbGL1
from src.schimdt.Solver import Solver


def _unit_problem(directory, loss, lambda_shrinkage, percent, datasetName, params, dbDir='../../../db'):
    log_path = directory + "/{}".format(datasetName)
    print("Working on: {}...".format(datasetName))
    fileType = fileTypeDict[datasetName]
    try:
        X, y = utils.set_up_xy(datasetName, fileType, dbDir)

        if loss == 'logit':
            f = LogisticLoss(X, y, log_path)
        elif loss == 'ls':
            f = LeastSquares(X, y, log_path)
        p = X.shape[1]
        num_of_groups = max(int(p * percent), 2)
        group = utils.gen_group(p, num_of_groups)
        lammax_path = f'../../../db/lammax-{datasetName}-{percent}.mat'
        Lip_path = f'../../../db/Lip-{datasetName}.mat'
        if os.path.exists(lammax_path):
            lammax = loadmat(lammax_path)["lammax"][0][0]
            print(f"loading lammax from: {lammax_path}")
        else:
            lammax = utils.lam_max(X, y, group, loss)
            savemat(lammax_path, {"lammax": lammax})
            print(f"save lammax to: {lammax_path}")
        Lambda = lammax * lambda_shrinkage
        # Lambda = utils.lam_max(X, y, group) * lambda_shrinkage
        r = GL1(Lambda=Lambda, group=group)
        prob = ProbGL1(f, r)
        solver = Solver(prob, params)
        if os.path.exists(Lip_path):
            L = loadmat(Lip_path)["L"][0][0]
            print(f"loading Lipschitz constant from: {Lip_path}")
        else:
            L = utils.estimate_lipschitz(X, loss)
            savemat(Lip_path, {"L": L})
            print(f"save Lipschitz constant to: {Lip_path}")
        info = solver.solve(alpha=1 / L)
        datasetid = "{}_{}_{}".format(datasetName, percent, lambda_shrinkage)
        info['datasetid'] = datasetid
        info_name = directory + "/{}_info.npy".format(datasetName)
        np.save(info_name, info)
    except FileNotFoundError:
        print(f"{datasetName} is not found. Skip!")


def runall(date, solver, loss, lambda_shrinkage, percent, datasets, params, dbDir='../../../db'):
    # create log directory
    directory = f"../log/{date}/{solver}/{loss}/{lambda_shrinkage}_{percent}_{''}_{''}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for datasetName in datasets:
        _unit_problem(directory, loss, lambda_shrinkage, percent, datasetName, params, dbDir)
