'''
File: logitbatch.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-04-06 01:27
Last Modified: 2021-04-06 02:08
--------------------------------------------
Description:
'''
import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import argparse
from src.params import params
from _batch import runall
import numpy as np

parser = argparse.ArgumentParser(description='Inexact Proximal Gradient bacth testing.')
parser.add_argument('--date', default='04_23_2021', type=str, help='experiment date')
parser.add_argument('--solver', default='negT', type=str, help='naive/schimdt/negT')
parser.add_argument('--loss', default='logit', type=str, help='ls/logit')
parser.add_argument('--lam_shrink', default=0.1, type=float, help='lambda shrink parameters')
parser.add_argument('--tol', default=1e-3, type=float, help='desired accuracy')
parser.add_argument('--t', default=1e-12, type=float, help='params for naive')
parser.add_argument('--safeguard_opt', default='none', type=str, help='param')
parser.add_argument('--safeguard_const', default=np.inf, type=float, help='param')
parser.add_argument('--schimdt_const', default=1.0, type=float, help='param')
args = parser.parse_args()

params['max_time'] = 2000
params['inexact_strategy'] = 'subgradient'
params['tol'] = args.tol
params['update_alpha_strategy'] = 'none'
params['t'] = -0.999
params['safeguard_opt'] = args.safeguard_opt
params['safeguard_const'] = args.safeguard_const
params['schimdt_const'] = args.schimdt_const
percents = [0.1, 0.2]

del params['warm_sampling']
del params['init_perturb']
del params['mode']
if args.solver == 'naive':
    del params['schimdt_const']
if args.solver == 'schimdt':
    del params['safeguard_opt']
    del params['safeguard_const']


if args.loss == 'logit':
    datasets = ["a9a", "australian", "breast_cancer", "german_numer",
                "ijcnn1", "ionosphere", "mushrooms", "splice", "sonar", "svmguide3", "w8a"]
else:
    datasets = ['abalone_scale', 'bodyfat_scale', 'cadata', 'cpusmall_scale',
                'housing_scale', 'pyrim_scale', 'YearPredictionMSD',
                'triazines_scale', 'virusShare']
for percent in percents:
    # local run
    # runall(args.date, 'naive', args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../db')
    # polyps run
    runall(args.date, args.solver, args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../../GroupFaRSA/db')
