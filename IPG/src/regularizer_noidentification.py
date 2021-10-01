'''
# File: regularizer.py
# Project: ipg
# Created Date: 2021-08-23 11:31
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-09-10 1:13
# Modified By: Yutong Dai yutongdai95@gmail.com
#
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''

import numpy as np
from numba import jit
from numba.typed import List
from scipy.sparse import csc_matrix, coo_matrix
import warnings


# def l2_norm(x):
#     return np.sqrt(np.dot(x.T, x))[0][0]
def l2_norm(x):
    return np.sqrt(np.sum(x * x))


def prox_primal(xk, uk, alphak, rxk):
    return (0.5 / alphak) * l2_norm(xk - uk) ** 2 + rxk


def prox_dual(y, uk, alphak):
    return -(alphak / 2 * l2_norm(y) ** 2 + uk.T @ y)


@jit(nopython=True, cache=True)
def _proj_norm_ball_jit(y, K, starts, ends, weights):
    projected_group = {}
    for i in range(K):
        start, end = starts[i], ends[i]
        y_Gi = y[start:end]
        norm_y_Gi = np.sqrt(np.dot(y_Gi.T, y_Gi))[0][0]
        if norm_y_Gi > weights[i]:
            y[start:end] = (weights[i] / norm_y_Gi) * y_Gi
            projected_group[i] = i
    return y, projected_group


"""
Overlapping Group L1 class
"""


class NatOG:
    def __init__(self, groups, penalty=1.0, weights=None):
        """
            groups: [[0,1,3,4], [0,2,4,5], [1,3,6]]
                index begins with 0
                all coordinates need to be included
                within each group the index needs to be sorted
        """
        if weights is not None:
            assert len(groups) == len(
                weights), "groups and weights should be of the same length"
        self.penalty = penalty
        self.K = len(groups)
        if weights is None:
            weights = np.array([np.sqrt(len(g)) for g in groups])
        self.weights = self.penalty * weights
        # group structure to matrix representation and
        self.lifted_dimension = 0
        self.groups_dict = {}
        groups_flattern = []
        group_size = []
        for (i, g) in enumerate(groups):
            groups_flattern += g
            self.lifted_dimension += len(g)
            group_size.append(len(g))
            self.groups_dict[i] = g
        # self.groups = np.array([np.array(g) for g in groups], dtype=object)
        self.groups = List()
        for g in groups:
            self.groups.append(np.array(g))
        # actual dimension
        p = max(groups_flattern) + 1
        # self.A = np.zeros((p, self.lifted_dimension))
        rows, cols = [], []
        for (colidx, rowidx) in enumerate(groups_flattern):
            rows.append(rowidx)
            cols.append(colidx)
        vals = [1.0] * len(rows)
        self.A = coo_matrix((vals, (rows, cols)),
                            shape=(p, self.lifted_dimension))
        self.A = self.A.tocsc()
        self.ATA = self.A.T @ self.A

        # relabel groups in lifted space
        self.starts, self.ends = [], []
        start = 0
        for i in group_size:
            self.starts.append(start)
            end = start + i
            self.ends.append(end)
            start = end
        self.starts, self.ends = np.array(self.starts), np.array(self.ends)

    def __str__(self):
        return("Overlapping Group L1")

    def func(self, x):
        return self._func_jit(x, self.groups, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _func_jit(x, groups, weights):
        ans = 0.0
        for i, g in enumerate(groups):
            xg = x[g]
            ans += np.sqrt(np.dot(xg.T, xg))[0][0] * weights[i]
        return ans

    ##############################################
    #    inexact proximal gradient calculation   #
    ##############################################

    def compute_inexact_proximal_gradient_update(self, xk, alphak, gradfxk, config, y_init, stepsize_init, **kwargs):
        """
            implement the fixed stepsize projected  gradient descent
        """
        uk = xk - alphak * gradfxk
        ATuk = self.A.T @ uk
        inexact_pg_computation = config['mainsolver']['inexact_pg_computation']
        if y_init is None:
            y_current = np.zeros((self.lifted_dimension, 1))
        else:
            assert y_init.shape[
                0] == self.lifted_dimension, f'y_init dimension:{y_init.shape[0]} mismacth with the desired lifted dimenson:{self.lifted_dimension}'
            y_current = y_init
        if stepsize_init is None:
            self.stepsize = 1 / alphak
        else:
            self.stepsize = stepsize_init
        self.inner_its = 0
        if not config['mainsolver']['exact_pg_computation']:
            if inexact_pg_computation == 'schimdt':
                # outter iteration counter begins with 0, therefore add 1
                k = kwargs['iteration'] + 1
                self.targap = config['inexactpg']['schimdt']['c'] / \
                    k**config['inexactpg']['schimdt']['delta']
                # self.targap = 1e-16
                # warnings.warn("Exact solve!")
            elif inexact_pg_computation == 'yd':
                ckplu1 = (np.sqrt(
                    6 / (1 + config['inexactpg']['yd']['gamma'] * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
            elif inexact_pg_computation == 'lee':
                primal_val_xk = prox_primal(
                    xk, uk, alphak, self.func(xk))
                gamma = config['inexactpg']['lee']['gamma']
            else:
                raise ValueError(
                    f"Unrecognized inexact_pg_computation value:{inexact_pg_computation}")
        else:
            self.targap = config['mainsolver']['exact_pg_computation_tol']

        dual_val_ycurrent = prox_dual(self.A @ y_current, uk, alphak)[0][0]
        grad_psi_ycurrent = (alphak * self.ATA @ y_current + ATuk)
        # print("===")
        # print(f"outer iter:{kwargs['iteration']}")
        # print("y_current:", y_current.T)
        # print("xk:", xk.T)
        # print("gradfxk", gradfxk.T)
        # print("alphak:", alphak, 'stepsize', self.stepsize)
        self.total_bak = 0
        while True:
            self.inner_its += 1
            # perform arc search to find suitable stepsize
            bak = 0
            if config['subsolver']['linesearch']:
                while True:
                    ytrial, projected_group = self._proj_norm_ball(
                        y_current - self.stepsize * grad_psi_ycurrent)
                    dual_val = prox_dual(self.A @ ytrial, uk, alphak)[0][0]
                    # print("projected_group", projected_group)
                    # taking negative because we are doing projected gradient descent with respect to the
                    # negative of the dual objective, namely psi function
                    LHS = -(dual_val - dual_val_ycurrent)
                    RHS = (config['linesearch']['eta'] *
                           (grad_psi_ycurrent.T @ (ytrial - y_current)))[0][0]
                    # if kwargs['iteration'] == 68:
                    #     print(
                    #         f"its:{kwargs['iteration']:3d} | LHS:{LHS:.4e} | RHS:{RHS:.4e} | LHS-RHS:{LHS-RHS:.4e}")
                    # polyps precision is low change 1e-16 to 1e-12
                    if (LHS <= RHS) or (np.abs(np.abs(LHS) - np.abs(RHS)) < 1e-15):
                        self.total_bak += bak
                        break
                    if self.stepsize < 1e-20:
                        # self.aoptim = 1e9
                        self.flag = 'smallstp'
                        # self.gap = 1e9
                        # self.targap = 1e9
                        self.total_bak += bak
                        # print("subsolver: small stepsize encountered!")
                        break
                        # return None, None, 1e9
                    self.stepsize *= config['linesearch']['xi']
                    bak += 1
            else:
                ytrial, projected_group = self._proj_norm_ball(
                    y_current - self.stepsize * grad_psi_ycurrent)
                dual_val = prox_dual(self.A @ ytrial, uk, alphak)[0][0]

            # get the primal approximate solution from the dual
            xtrial = alphak * (self.A @ ytrial) + uk
            # print("xtrial", xtrial.T)
            # print("xtrial_proj", xtrial_proj.T)
            ######################### check for termination ###############################
            # # first check the projected primal
            # rxtrial_proj = self.func(xtrial_proj)
            # primal_val_proj = prox_primal(
            #     xtrial_proj, uk, alphak, rxtrial_proj)
            # gap = (primal_val_proj - dual_val)
            # if not config['mainsolver']['exact_pg_computation']:
            #     if inexact_pg_computation == 'yd':
            #         self.targap = ckplu1 * l2_norm(xtrial_proj - xk) ** 2
            #     elif inexact_pg_computation == 'lee':
            #         # this is correct and I verified at 09/14/2021 using (12) in paper
            #         self.targap = gamma * (primal_val_xk - dual_val)
            #     elif inexact_pg_computation == 'schimdt':
            #         pass
            #     else:
            #         raise ValueError(
            #             f"Unrecognized inexact_pg_computation value:{inexact_pg_computation}")
            # if gap < self.targap:
            #     xtrial = xtrial_proj
            #     self.flag = 'desired'
            #     self.rxtrial = rxtrial_proj
            #     break
            # then check the un-projected primal
            rxtrial = self.func(xtrial)
            primal_val = prox_primal(xtrial, uk, alphak, rxtrial)
            gap = (primal_val - dual_val)
            if not config['mainsolver']['exact_pg_computation']:
                if inexact_pg_computation == 'yd':
                    self.targap = ckplu1 * l2_norm(xtrial - xk) ** 2
                elif inexact_pg_computation == 'lee':
                    # this is correct and I verified at 09/14/2021 using (12) in paper
                    self.targap = gamma * (primal_val_xk - dual_val)
                elif inexact_pg_computation == 'schimdt':
                    pass
                else:
                    raise ValueError(
                        f"Unrecognized inexact_pg_computation value:{inexact_pg_computation}")
            if gap < self.targap:
                self.flag = 'desired'
                self.rxtrial = rxtrial
                break
            if self.inner_its > config['subsolver']['iteration_limits']:
                self.flag = 'maxiter'
                # attemp a correction step
                x_correction = self.correction_step(xtrial, kwargs['xref'])
                rx_correction = self.func(x_correction)
                primal_val_correction = prox_primal(
                    x_correction, uk, alphak, rx_correction)
                gap_corrected = primal_val_correction - dual_val
                if primal_val_correction <= primal_val:
                    xtrial = x_correction
                    gap = gap_corrected
                    rxtrial = rx_correction
                    self.flag = 'correct'
                self.rxtrial = rxtrial
                break
            # proceed to the next iteration
            y_current = ytrial
            grad_psi_ycurrent = (alphak * self.ATA @ y_current + ATuk)
            dual_val_ycurrent = dual_val
            # if no backtracking is performed, increase the stepsize for the next iteration
            if bak == 0:
                self.stepsize *= config['linesearch']['beta']
        # post-processing
        self.gap = gap
        if not config['mainsolver']['exact_pg_computation'] and inexact_pg_computation == 'yd' and self.flag != 'maxiter':
            self.aoptim = np.sqrt(self.targap / ckplu1)
        else:
            self.aoptim = l2_norm(xtrial - xk)
        self.xtrial = xtrial
        return xtrial, ytrial, self.aoptim

    def _proj_norm_ball(self, y):
        return _proj_norm_ball_jit(y, self.K, self.starts, self.ends, self.weights)

    def print_header(self, config):
        # printlevel == 1
        header = "  aoptim   its.   Flag "
        if config['mainsolver']['print_level'] == 2:
            header += " Stepsize  baks    Gap       tarGap "
            header += "   #pz  #pnz |"
        else:
            header += "|"
        return header

    def print_iteration(self, config):
        content = f" {self.aoptim:.3e} {self.inner_its:4d} {self.flag}"
        if config['mainsolver']['print_level'] == 2:
            content += f" {self.stepsize:.3e} {self.total_bak:4d} {self.gap:+.3e} {self.targap:+.3e}"
            nnz, nz = self._get_group_structure(self.xtrial)
            content += f" {nz:4d}  {nnz:4d} |"
        else:
            content += "|"
        return content

    def _get_group_structure(self, X):
        return self._get_group_structure_jit(X, self.K, self.groups)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_group_structure_jit(X, K, groups):
        nz = 0
        for g in groups:
            X_Gi = X[g]
            if (np.sum(np.abs(X_Gi)) == 0):
                nz += 1
        nnz = K - nz
        return nnz, nz

    def correction_step(self, x, xref):
        return self._correction_step_jit(x, xref, self.groups)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _correction_step_jit(x, xref, groups):
        x_corrected = x.copy()
        for g in groups:
            if np.sum(np.abs(xref[g])) == 0:
                x_corrected[g] = 0.0
        return x_corrected
