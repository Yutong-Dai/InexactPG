'''
# File: runall.py
# Project: logit
# Created Date: 2021-09-09 11:59
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-09-10 12:26
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
import src.utils as utils
from src.params import *
import yaml
import platform
import argparse
import numpy as np


def _unit_problem(directory, inexact_type, loss, lam_shrink, group_size, overlap_ratio, datasetName, dbDir, save_ckpt,
                  milestone, date, config):
    log_path = directory + "/{}".format(datasetName)
    print("Working on: {}... | inexact_type: {}".format(
        datasetName, inexact_type), flush=True)
    fileType = fileTypeDict[datasetName]
    FileExist = True
    try:
        X, y = utils.set_up_xy(datasetName, fileType, dbDir)
    except FileNotFoundError:
        dbDir += '_big'
        try:
            X, y = utils.set_up_xy(datasetName, fileType, dbDir)
        except FileNotFoundError:
            print(f"{datasetName} is not found. Skip!")
            FileExist = False
    if FileExist:
        # set up f
        if loss == 'logit':
            f = LogisticLoss(X, y, log_path)
        elif loss == 'ls':
            f = LeastSquares(X, y, log_path)

        # setting up groups
        p = X.shape[1]
        grp_size = min(p // 2, group_size)
        generator = utils.GenOverlapGroup(
            p, grp_size=grp_size, overlap_ratio=overlap_ratio)
        groups, starts, ends = generator.get_group()

        # calculate lammax and Lipschitz constants
        lammax_path = f'{dbDir}/lammax-{datasetName}-{grp_size}-{overlap_ratio}.mat'
        # Lip_path = f'{dbDir}/Lip-{datasetName}.mat'
        if os.path.exists(lammax_path):
            lammax = loadmat(lammax_path)["lammax"][0][0]
            print(f"loading lammax from: {lammax_path}")
        else:
            lammax = utils.lam_max(X, y, starts, ends, loss)
            savemat(lammax_path, {"lammax": lammax})
            print(f"save lammax to: {lammax_path}")

        # setting up the regularizer
        r = NatOG(penalty=lammax * lam_shrink, groups=groups, weights=None)
        save_ckpt_id = {'date': date, 'loss': loss,
                        "lam_shrink": lam_shrink, "grp_size": group_size, "overlap_ratio": overlap_ratio}
        solver = IpgSolver(f, r, config)

        info = solver.solve(alpha_init=1.0, save_ckpt=save_ckpt,
                            save_ckpt_id=save_ckpt_id, milestone=milestone)
        print(f"time:{info['time']:.3e} | its: {info['iteration']:4d} | subits:{info['subits']:5d} | F:{info['F']:.3e} | nnz:{info['nnz']:4d} | nz:{info['nz']:4d}")
        info_path = directory + "/{}_info.npy".format(datasetName)
        np.save(info_path, info)


def runall(date, inexact_type, loss, lam_shrink, group_size, overlap_ratio, datasets, dbDir, config, save_ckpt,
           milestone):
    directory = f"./log/{date}/{inexact_type}/{loss}/logfile/{lam_shrink}_{group_size}_{overlap_ratio}"
    if inexact_type == 'exact':
        directory += "_empty"
    elif inexact_type == 'schimdt':
        directory += f"_{config['inexactpg'][inexact_type]['c']}"
    elif inexact_type == 'lee':
        directory += f"_{config['inexactpg'][inexact_type]['gamma']}"
    elif inexact_type == 'yd':
        directory += f"_{config['inexactpg'][inexact_type]['gamma']}"
    else:
        raise ValueError(f"Unrecognized inexact_type:{inexact_type}!")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for datasetName in datasets:
        _unit_problem(directory, inexact_type, loss, lam_shrink,
                      group_size, overlap_ratio, datasetName, dbDir, save_ckpt,
                      milestone, date, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inexact Proximal Gradient bacth testing.')
    parser.add_argument('--date', default='09_10_2021',
                        type=str, help='experiment date')
    parser.add_argument('--loss', default='logit', type=str, help='ls/logit')
    parser.add_argument('--lam_shrink', default=0.1, type=float,
                        help='lambda shrink parameters')
    parser.add_argument('--group_size', default=10, type=int,
                        help='number of variables per group')
    parser.add_argument('--overlap_ratio', default=0.1,
                        type=float, help='overlap ratio for each groups')
    parser.add_argument('--tol', default=1e-6, type=float,
                        help='desired accuracy')
    parser.add_argument('--max_time', default=7200,
                        type=int, help='max time in seconds')
    parser.add_argument('--inexact_type', default='yd',
                        type=str, help='exact/schimdt/lee/yd')
    parser.add_argument('--largedb', default=False, type=lambda x: (
        str(x).lower() in ['true', '1', 'yes']), help='test large db')
    parser.add_argument('--c', default=1.0,
                        type=float, help='params for inexact_type schimdt')
    parser.add_argument('--gamma_lee', default=0.5, type=float,
                        help='params for inexact_type lee')
    parser.add_argument('--gamma_yd', default=0.1, type=float,
                        help='params for inexact_type yd')

    args = parser.parse_args()
    with open('../src/config.yaml', "r") as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    if args.inexact_type == 'exact':
        config['mainsolver']['exact_pg_computation'] = True
    else:
        config['mainsolver']['exact_pg_computation'] = False
        config['mainsolver']['inexact_pg_computation'] = args.inexact_type
        if args.inexact_type == 'schimdt':
            config['inexactpg'][args.inexact_type]['c'] = args.c
        elif args.inexact_type == 'lee':
            config['inexactpg'][args.inexact_type]['gamma'] = args.gamma_lee
        elif args.inexact_type == 'yd':
            config['inexactpg'][args.inexact_type]['gamma'] = args.gamma_yd
        else:
            raise ValueError(f"Unrecognized inexact_type:{args.inexact_type}!")

    config['mainsolver']['accuracy'] = args.tol
    config['mainsolver']['time_limits'] = args.max_time
    save_ckpt = True
    milestone = [1e-3, 1e-4, 1e-5]
    if args.loss == 'logit':
        if not args.largedb:
            if platform.platform() == 'Darwin-16.7.0-x86_64-i386-64bit':
                datasets = ["w8a"]
            else:
                datasets = ["a9a", "colon_cancer", "duke",
                            "leu", "mushrooms", "w8a"]
        else:
            datasets = ["madelon", "gisette", "rcv1", "real-sim", "news20"]
    else:
        datasets = ['abalone_scale', 'bodyfat_scale', 'cadata', 'cpusmall_scale',
                    'housing_scale', 'pyrim_scale', 'YearPredictionMSD',
                    'triazines_scale', 'virusShare']

    # local run
    if platform.platform() == 'Darwin-16.7.0-x86_64-i386-64bit':
        runall(args.date, args.inexact_type, args.loss, args.lam_shrink, args.group_size, args.overlap_ratio,
               datasets, '/Users/ym/Documents/GroupFaRSA/db', config, save_ckpt, milestone)
    else:
        # polyps run
        runall(args.date, args.inexact_type, args.loss, args.lam_shrink, args.group_size, args.overlap_ratio,
               datasets, '../../../GroupFaRSA/db', config, save_ckpt, milestone)
