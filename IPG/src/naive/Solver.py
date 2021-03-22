'''
File: Solver.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:44
Last Modified: 2021-03-22 18:35
--------------------------------------------
Description:
'''
import sys
import os
from llvmlite.ir.values import Value

from numpy.lib import utils
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
from src.params import params
import src.utils as utils
import printUtils


class Solver:
    def __init__(self, prob, params):
        self.prob = prob
        self.__dict__.update(params)
        self.version = "0.1 (2021-03-22)"

    def set_init_alpha(self, x):
        s = 1e-2
        _ = self.prob.funcf(x)
        gradfx = self.prob.gradf(x)
        y = x - s * gradfx
        while True:
            if utils.l2_norm(y - x) > 1e-8:
                _ = self.prob.funcf(y)
                gradfy = self.prob.gradf(y)
                alpha = utils.l2_norm(x - y) / (1 * utils.l2_norm(gradfx - gradfy))
                break
            else:
                s *= 10
                y = x - s * gradfx
        return alpha

    def update(self, x, gradfx, alpha, fvalx, rvalx):
        self.bak = 0
        self.ipg_nnz, self.ipg_nz = None, None
        while True:
            xtrial = self.prob.ipg(x, gradfx, alpha, self.inexact_strategy,
                                   init_perturb=self.init_perturb,
                                   t=self.t, mode=self.mode, seed=0)
            # optional: get sparsity sturctures.
            if self.prob.nz is not None:
                self.pg_nz, self.pg_nnz = self.prob.nz, self.prob.nnz
            if self.pg_nz is not None:
                self.ipg_nnz = utils.get_group_structure(xtrial, self.prob.K, self.prob.starts, self.prob.ends)
                self.ipg_nz = self.prob.K - self.ipg_nnz

            fval_xtrial = self.prob.funcf(xtrial)
            rval_xtrial = self.prob.funcr(xtrial)
            d = xtrial - x
            d_norm = utils.l2_norm(d)
            d_norm_sq = d_norm ** 2
            epsilon = self.prob.ck * d_norm_sq
            LHS = fval_xtrial + rval_xtrial - fvalx - rvalx
            RHS = self.eta * (d_norm_sq / alpha - np.sqrt(2 * epsilon / alpha) * d_norm - epsilon)

            if LHS <= RHS:
                return xtrial, fval_xtrial, rval_xtrial, alpha
            if self.bak == self.maxbak:
                # line search failed
                return None, None, None, None, None
            if self.update_alpha_strategy == 'frac':
                alpha *= self.zeta
            elif self.update_alpha_strategy == 'model':
                dirder = np.dot(gradfx.T, d)[0][0]
                actual_decrease = fval_xtrial - fvalx
                L_local = 2 * (actual_decrease - dirder) / d_norm_sq
                alpha = max(alpha * self.zeta, 1 / L_local)
            else:
                raise ValueError(f'Invalid update_alpha_strategy: {self.update_alpha_strategy}')
            self.bak += 1

    def solve(self, x=None):
        if not x:
            x = np.zeros((self.prob.p, 1))
