'''
File: batch.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-06-21 00:46
Last Modified: 2021-04-06 02:09
--------------------------------------------
Description:
'''

import sys
from scipy.io import savemat, loadmat
import os
sys.path.append("../..")

import numpy as np

import src.utils as utils
from src.params import fileTypeDict
from src.regularizer import LatentOG
from src.lossfunction import LogisticLoss, LeastSquares
from src.latentOG.Problem import ProbLatentOG
from src.latentOG.Solver import Solver


def _unit_problem(directory, inexact_type, loss, lambda_shrinkage, group_size, overlap_ratio, datasetName, params, dbDir='../../../db'):
    log_path = directory + "/{}".format(datasetName)
    params['projectedGD']['stepsize'] = 1.0
    print("Working on: {}... | inexact_type: {}".format(datasetName, inexact_type), flush=True)
    print(f"Initial stepsize:{params['projectedGD']['stepsize']}", flush=True)
    fileType = fileTypeDict[datasetName]
    try:
        X, y = utils.set_up_xy(datasetName, fileType, dbDir)

        if loss == 'logit':
            f = LogisticLoss(X, y, log_path)
        elif loss == 'ls':
            f = LeastSquares(X, y, log_path)
        p = X.shape[1]
        group_size = min(p // 2, group_size)
        generator = utils.GenOverlapGroup(p, grp_size=group_size, overlap_ratio=overlap_ratio)
        starts, ends = generator.get_group()
        # calculate lammax and Lipschitz constants
        lammax_path = f'{dbDir}/lammax-{datasetName}-{group_size}-{overlap_ratio}.mat'
        Lip_path = f'{dbDir}/Lip-{datasetName}.mat'
        if os.path.exists(lammax_path):
            lammax = loadmat(lammax_path)["lammax"][0][0]
            print(f"loading lammax from: {lammax_path}")
        else:
            lammax = utils.lam_max(X, y, starts, ends, loss)
            savemat(lammax_path, {"lammax": lammax})
            print(f"save lammax to: {lammax_path}")

        if os.path.exists(Lip_path):
            L = loadmat(Lip_path)["L"][0][0]
            print(f"loading Lipschitz constant from: {Lip_path}")
        else:
            L = utils.estimate_lipschitz(X, loss)
            savemat(Lip_path, {"L": L})
            print(f"save Lipschitz constant to: {Lip_path}")
        r = LatentOG(Lambda=lammax * 0.8, dim=p, starts=starts, ends=ends)
        prob = ProbLatentOG(f, r)
        solver = Solver(prob, params)
        info = solver.solve(alpha=1 / L, explore=True)
        datasetid = "{}_{}_{}_{}".format(datasetName, lambda_shrinkage, group_size, overlap_ratio)
        info['datasetid'] = datasetid
        info_name = directory + "/{}_info.npy".format(datasetName)
        np.save(info_name, info)
    except FileNotFoundError:
        print(f"{datasetName} is not found. Skip!")


def runall(date, inexact_type, loss, lambda_shrinkage, group_size, overlap_ratio, datasets, params, dbDir='../../../db'):
    # create log directory
    directory = f"../log/{date}/{inexact_type}/{loss}/{params['subsolver']}_{params['warm_start']}_{lambda_shrinkage}_{group_size}_{overlap_ratio}"
    if inexact_type == 1:
        directory += f"_{params['gamma1']}_empty"
    elif inexact_type == 2:
        directory += f"_{params['gamma2']}_{params['nu']}"
    else:
        directory += f"_{params['delta']}_{params['schimdt_const']}"

    if not os.path.exists(directory):
        os.makedirs(directory)
    for datasetName in datasets:
        _unit_problem(directory, inexact_type, loss, lambda_shrinkage, group_size, overlap_ratio, datasetName, params, dbDir)
