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
parser.add_argument('--date', default='07_09_2021', type=str, help='experiment date')
parser.add_argument('--loss', default='logit', type=str, help='ls/logit')
parser.add_argument('--lam_shrink', default=0.1, type=float, help='lambda shrink parameters')
parser.add_argument('--group_size', default=10, type=int, help='number of variables per group')
parser.add_argument('--overlap_ratio', default=0.1, type=float, help='overlap ratio for each groups')
parser.add_argument('--tol', default=1e-5, type=float, help='desired accuracy')
parser.add_argument('--max_time', default=7200, type=int, help='max time in seconds')
parser.add_argument('--inexact_type', default=1, type=int, help='1/2/3')
parser.add_argument('--subsolver', default='projectedGD', type=str, help='desired accuracy')
parser.add_argument('--warm_start', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='warm start for subsolver')
parser.add_argument('--largedb', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='test large db')
parser.add_argument('--gamma1', default=1e-12, type=float, help='params for inexact_type 1')
parser.add_argument('--gamma2', default=1e-12, type=float, help='params for inexact_type 2')
parser.add_argument('--nu', default=0.5, type=float, help='params for inexact_type 2')
parser.add_argument('--delta', default=3, type=float, help='params for inexact_type 3')
parser.add_argument('--schimdt_const', default=1.0, type=float, help='params for inexact_type 3')
args = parser.parse_args()

params['tol'] = args.tol
params['max_time'] = args.max_time
params['inexact_type'] = args.inexact_type
params['subsolver'] = args.subsolver
params['gamma1'] = args.gamma1
params['gamma2'] = args.gamma2
params['nu'] = args.nu
params['delta'] = args.delta
params['schimdt_const'] = args.schimdt_const
params['subsolver_verbose'] = False
params['scale_alpha'] = False
params['ckpt_tol'] = 1e-4
params['ckpt'] = True

if args.loss == 'logit':
    if not args.largedb:
        datasets = ["a9a", "colon_cancer", "duke",
                    "leu", "mushrooms", "w8a"]
    else:
        datasets = ["rcv1", "real-sim", "news20", "gisette", "madelon"]
else:
    datasets = ['abalone_scale', 'bodyfat_scale', 'cadata', 'cpusmall_scale',
                'housing_scale', 'pyrim_scale', 'YearPredictionMSD',
                'triazines_scale', 'virusShare']

# local run
# runall(args.date, 'naive', args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../db')
# polyps run
runall(args.date, args.inexact_type, args.loss, args.lam_shrink, args.group_size, args.overlap_ratio,
       datasets, params, dbDir='../../../../GroupFaRSA/db')

# print(params['max_time'])
# print(datasets)
