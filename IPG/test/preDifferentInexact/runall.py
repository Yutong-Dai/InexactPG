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

parser = argparse.ArgumentParser(description='Inexact Proximal Gradient bacth testing.')
parser.add_argument('--date', default='04_10_2021', type=str, help='experiment date')
parser.add_argument('--solver', default='naive', type=str, help='naive/schimdt')
parser.add_argument('--loss', default='logit', type=str, help='ls/logit')
parser.add_argument('--lam_shrink', default=0.1, type=float, help='lambda shrink parameters')
parser.add_argument('--init_perturb', default=1e1, type=float, help='1e1: inexact solve; 0 exact solve')
parser.add_argument('--safeguard_opt', default='schimdt', type=str, help='param')
parser.add_argument('--safeguard_const', default=1.0, type=float, help='param')
args = parser.parse_args()

params['init_perturb'] = args.init_perturb
params['tol'] = 1e-6
params['update_alpha_strategy'] = 'none'
params['t'] = 1e-12
params['safeguard_opt'] = args.safeguard_opt
params['safeguard_const'] = args.safeguard_const
percents = [0.1, 0.3]

if args.loss == 'logit':
    datasets = ["a9a", "australian", "breast_cancer", "german_numer",
                "ijcnn1", "ionosphere", "splice", "svmguide3", "w8a"]
else:
    datasets = ['abalone_scale', 'bodyfat_scale', 'cpusmall_scale',
                'housing_scale', 'pyrim_scale',
                'YearPredictionMSD', 'triazines_scale', 'virusShare']
for percent in percents:
    # local run
    # runall(args.date, 'naive', args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../db')
    # polyps run
    runall(args.date, args.solver, args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../../GroupFaRSA/db')
