'''
File: logitbatch.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-04-06 01:27
Last Modified: 2021-04-06 02:21
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
parser.add_argument('--date', default='04_06_2021', type=str, help='lambda shrink parameters')
parser.add_argument('--loss', default='logit', type=str, help='lambda shrink parameters')
parser.add_argument('--lam_shrink', default=0.1, type=float, help='lambda shrink parameters')
args = parser.parse_args()

params['init_perturb'] = 1e2
params['tol'] = 1e-6
params['update_alpha_strategy'] = 'none'
percents = [0.1, 0.3, 0.5]

if args.loss == 'logit':
    datasets = ["a9a", "australian", "breast_cancer", "colon_cancer",
                "duke", "german_numer", 'heart', "ijcnn1", "ionosphere", "leu",
                "mushrooms", "phishing", "splice", "sonar", "svmguide1", "svmguide3", "w8a"
                ]
else:
    datasets = ['abalone_scale', 'bodyfat_scale', 'cadata', 'cpusmall_scale', 'eunite2001',
                'housing_scale', 'mg_scale', 'mpg_scale', 'pyrim_scale', 'space_ga_scale', 'blogData_train',
                'UJIIndoorLoc', 'driftData', 'YearPredictionMSD', 'triazines_scale', 'virusShare']
for percent in percents:
    # runall(args.date, 'schimdtPrescribe', args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../db')
    # polyps
    runall(args.date, 'schimdtPrescribe', args.loss, args.lam_shrink, percent, datasets, params, dbDir='../../../../GroupFaRSA/db')
