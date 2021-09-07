'''
# File: regularizer.py
# Project: ipg
# Created Date: 2021-08-23 11:31
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-08-31 12:40
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


def l2_norm(x):
    return np.sqrt(np.dot(x.T, x))[0][0]


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
        assert len(groups) == len(
            weights), "groups and weights should be of the same lengthg"
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
        self.A = np.zeros((p, self.lifted_dimension))
        rows, cols = [], []
        for (colidx, rowidx) in enumerate(groups_flattern):
            rows.append(rowidx)
            cols.append(colidx)
        vals = [1.0] * len(rows)
        self.A = coo_matrix((vals, (rows, cols)),
                            shape=(p, self.lifted_dimension))
        self.A = self.A.tocsc()

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
    @jit(nopython=True)
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
        inexact_strategy = config['mainsolver']['inexact_strategy']
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
        if inexact_strategy == 'schimdt':
            # outter iteration counter, begin with 0, therefore add 1
            k = kwargs['iteration'] + 1
            self.targap = config['inexactpg']['schimdt']['c'] / \
                (k + 1)**config['inexactpg']['schimdt']['delta']
            # self.targap = 1e-16
            # warnings.warn("Exact solve!")
            while True:
                self.inner_its += 1
                ytrial, projected_group = self._proj_norm_ball(
                    y_current - self.stepsize * (alphak * self.A.T @ self.A @ y_current + self.A.T @ uk))
                xtrial = alphak * (self.A @ ytrial) + uk
                xtrial_proj = xtrial.copy()
                for i in range(self.K):
                    if i not in projected_group:
                        xtrial_proj[self.groups_dict[i]] = 0.0
                # check for termination
                dual_val = prox_dual(self.A @ ytrial, uk, alphak)
                primal_val_proj = prox_primal(
                    xtrial_proj, uk, alphak, self.func(xtrial_proj))
                gap = (primal_val_proj - dual_val)[0][0]
                if gap < self.targap:
                    xtrial = xtrial_proj
                    self.flag = 'desired'
                    break
                primal_val = prox_primal(
                    xtrial, uk, alphak, self.func(xtrial))
                gap = (primal_val - dual_val)[0][0]
                if gap < self.targap:
                    self.flag = 'desired'
                    break
                if self.inner_its > config['subsolver']['iteration_limits']:
                    self.flag = 'maxiters'
                    break
                y_current = ytrial
            self.gap = gap
            self.aoptim = l2_norm(xtrial - xk)
            self.xtrial = xtrial
            return xtrial, ytrial, self.aoptim

    def _proj_norm_ball(self, y):
        return _proj_norm_ball_jit(y, self.K, self.starts, self.ends, self.weights)

    def print_header(self, config):
        # printlevel == 1
        header = "  aoptim   its.   Flag "
        if config['mainsolver']['print_level'] == 2:
            header += " Stepsize     Gap       tarGap "
            header += "   #az  #anz |"
        else:
            header += "|"
        return header

    def print_iteration(self, config):
        content = f" {self.aoptim:.3e} {self.inner_its:4d} {self.flag}"
        if config['mainsolver']['print_level'] == 2:
            content += f" {self.stepsize:.3e} {self.gap:+.3e} {self.targap:+.3e}"
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


"""
Tree ovelapping Group L1 class
"""
try:
    import spams
    SPAM_EXISTS = True
except ModuleNotFoundError:
    SPAM_EXISTS = False

if SPAM_EXISTS:
    class TreeOG:
        def __init__(self, groups, tree, penalty, weights=None):
            """
                # Example 1 of tree structure
                # tree structured groups:
                # g1= {0 1 2 3 4 5 6 7 8 9}
                # g2= {2 3 4}
                # g3= {5 6 7 8 9}
                own_variables =  np.array([0,2,5],dtype=np.int32) # pointer to the first variable of each group
                N_own_variables =  np.array([2,3,5],dtype=np.int32) # number of "root" variables in each group
                # (variables that are in a group, but not in its descendants).
                # for instance root(g1)={0,1}, root(g2)={2 3 4}, root(g3)={5 6 7 8 9}
                eta_g = np.array([1,30,1],dtype=np.float64) # weights for each group, they should be non-zero to use fenchel duality
                groups = np.asfortranarray([[0,0,0],
                                            [1,0,0],
                                            [1,0,0]],dtype = np.bool)
                # first group should always be the root of the tree
                # non-zero entriees mean inclusion relation ship, here g2 is a children of g1,
                # g3 is a children of g1
                groups = csc_matrix(groups,dtype=np.bool)
                tree = {'eta_g': eta_g,'groups' : groups,'own_variables' : own_variables,
                        'N_own_variables' : N_own_variables}
            """
            assert len(groups) == len(
                weights), "groups and weights should be of the same length"
            self.penalty = penalty
            self.tree = tree
            self.K = len(groups)
            if weights is None:
                weights = np.array([np.sqrt(len(g)) for g in groups])
            self.weights = self.penalty * weights
            self.groups = List()
            for g in groups:
                self.groups.append(np.array(g))

        def __str__(self):
            return("Overlapping Tree Group L1")

        def func(self, x):
            return self._func_jit(x, self.groups, self.weights)

        @staticmethod
        @jit(nopython=True)
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
            param = {'numThreads': -1, 'verbose': False, 'pos': False,
                     'intercept': False, 'lambda1': alphak, 'regul': 'tree-l2'}
            uk = xk - alphak * gradfxk
            xtrial = spams.proximalTree(uk, self.tree, False, **param)
            self.xtrial = xtrial
            self.aoptim = l2_norm(xtrial - xk)
            return xtrial, None, self.aoptim

        def print_header(self, config):
            # printlevel == 1
            header = "  aoptim  "
            if config['mainsolver']['print_level'] == 2:
                header += "   #az  #anz |"
            else:
                header += "|"
            return header

        def print_iteration(self, config):
            content = f" {self.aoptim:.3e}"
            if config['mainsolver']['print_level'] == 2:
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


"""
None Overlapping Group L1 class
"""


class GL1:
    def __init__(self, groups, penalty=None, weights=None):
        """
        !!Warning: need `groups` be ordered in a consecutive manner, i.e.,
        groups: array([1., 1., 1., 2., 2., 2., 3., 3., 3., 3.])
        Then:
        unique_groups: array([1., 2., 3.])
        group_frequency: array([3, 3, 4]))
        """
        self.penalty = penalty
        self.unique_groups, self.group_frequency = np.unique(
            groups, return_counts=True)
        if weights is not None:
            self.weights = weights
        else:
            if penalty is not None:
                self.weights = penalty * np.sqrt(self.group_frequency)
            else:
                raise ValueError("Initialization failed!")
        self.K = len(self.unique_groups)
        self.group_size = -1 * np.ones(self.K)
        p = groups.shape[0]
        full_index = np.arange(p)
        starts = []
        ends = []
        for i in range(self.K):
            G_i = full_index[np.where(groups == self.unique_groups[i])]
            # record the `start` and `end` indices of the group G_i to avoid fancy indexing innumpy
            # in the example above, the start index and end index for G_1 is 0 and 2 respectively
            # since python `start:end` will include `start` and exclude `end`, so we will add 1 to the `end`
            # so the G_i-th block of X is indexed by X[start:end]
            start, end = min(G_i), max(G_i) + 1
            starts.append(start)
            ends.append(end)
            self.group_size[i] = end - start
        # wrap as np.array for jit compile purpose
        self.starts = np.array(starts)
        self.ends = np.array(ends)

    def __str__(self):
        return("Group L1")

    def func(self, X):
        """
            X here is not the data matrix but the variable instead
        """
        return self._func_jit(X, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True)
    def _func_jit(X, K, starts, ends, weights):
        fval = 0.0
        for i in range(K):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            # don't call l2_norm for jit to complie
            fval += weights[i] * np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
        return fval

    def grad(self, X):
        """
            compute the gradient. If evaluate at the group whose value is 0, then
            return np.inf for that group
        """
        return self._grad_jit(X, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True)
    def _grad_jit(X, K, starts, ends, weights):
        grad = np.full(X.shape, np.inf)
        for i in range(K):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            norm_XG_i = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            if (np.abs(norm_XG_i) > 1e-15):
                grad[start:end] = (weights[i] / norm_XG_i) * XG_i
        return grad

    def _prepare_hv_data(self, X, subgroup_index):
        """
        make sure the groups in subgroup_index are non-zero
        """
        self.hv_data = {}
        start = 0
        for i in subgroup_index:
            start_x, end_x = self.starts[i], self.ends[i]
            XG_i = X[start_x:end_x]
            XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            end = start + end_x - start_x
            self.hv_data[i] = {}
            self.hv_data[i]['XG_i'] = XG_i
            self.hv_data[i]['XG_i_norm'] = XG_i_norm
            self.hv_data[i]['start'] = start
            self.hv_data[i]['end'] = end
            self.hv_data[i]['XG_i_norm_cubic'] = XG_i_norm**3
            start = end

    def hessian_vector_product_fast(self, v, subgroup_index):
        """
        call _prepare_hv_data before call hessian_vector_product_fast
        """
        hv = np.empty_like(v)
        for i in subgroup_index:
            start = self.hv_data[i]['start']
            end = self.hv_data[i]['end']
            vi = v[start:end]
            temp = np.matmul(self.hv_data[i]['XG_i'].T, vi)
            hv[start:end] = self.weights[i] * (1 / self.hv_data[i]['XG_i_norm'] * vi -
                                               (temp / self.hv_data[i]['XG_i_norm_cubic']) *
                                               self.hv_data[i]['XG_i'])
        return hv

    def _dual_norm(self, y):
        """
            compute the dual of r(x), which is r(y): max ||y_g||/lambda_g
            reference: https://jmlr.org/papers/volume18/16-577/16-577.pdf section 5.2
        """
        return self._dual_norm_jit(y, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True)
    def _dual_norm_jit(y, K, starts, ends, weights):
        max_group_norm = 0.0
        for i in range(K):
            start, end = starts[i], ends[i]
            yG_i = y[start:end]
            temp_i = (np.sqrt(np.dot(yG_i.T, yG_i))[0][0]) / weights[i]
            max_group_norm = max(max_group_norm, temp_i)
        return max_group_norm

    ##############################################
    #      exact proximal gradient calculation   #
    ##############################################
    def compute_proximal_gradient_update(self, xk, alphak, gradfxk):
        self.prox, self.zeroGroup, self.nonZeroGroup = self._compute_proximal_gradient_update_jit(xk, alphak, gradfxk, self.starts,
                                                                                                  self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_proximal_gradient_update_jit(X, alpha, gradf, starts, ends, weights):
        proximal = np.zeros_like(X)
        nonZeroGroup = []
        zeroGroup = []
        for i in range(len(starts)):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            gradfG_i = gradf[start:end]
            gradient_step = XG_i - alpha * gradfG_i
            gradient_step_norm = np.sqrt(
                np.dot(gradient_step.T, gradient_step))[0][0]
            if gradient_step_norm != 0:
                temp = 1 - ((weights[i] * alpha) / gradient_step_norm)
            else:
                temp = -1
            if temp > 0:
                nonZeroGroup.append(i)
            else:
                zeroGroup.append(i)
            proximal[start:end] = max(temp, 0) * gradient_step
        return proximal, len(zeroGroup), len(nonZeroGroup)

    ##############################################
    #    inexact proximal gradient calculation   #
    ##############################################
    def compute_inexact_proximal_gradient_update(self, xk, alphak, gradfxk, config, y_init, stepsize_init, **kwargs):
        """
            implement the fixed stepsize projected  gradient descent
        """
        uk = xk - alphak * gradfxk
        inexact_strategy = config['mainsolver']['inexact_strategy']
        if y_init is None:
            y_current = np.zeros_like(uk)
        else:
            y_current = y_init
        if stepsize_init is None:
            self.stepsize = 1 / alphak
        else:
            self.stepsize = stepsize_init
        self.inner_its = 0
        if config['subsolver']['compute_exactpg']:
            self.compute_proximal_gradient_update(xk, alphak, gradfxk)
            self.optim = l2_norm(self.prox - xk)
        if inexact_strategy == 'schimdt':
            # outter iteration counter, begin with 0, therefore add 1
            k = kwargs['iteration'] + 1
            self.targap = config['inexactpg']['schimdt']['c'] / \
                (k + 1)**config['inexactpg']['schimdt']['delta']
            while True:
                self.inner_its += 1
                ytrial, projected_group = self._proj_norm_ball(
                    y_current - self.stepsize * (alphak * y_current + uk))
                xtrial = alphak * ytrial + uk
                xtrial_proj = xtrial.copy()
                for i in range(self.K):
                    if i not in projected_group:
                        start, end = self.starts[i], self.ends[i]
                        xtrial_proj[start:end] = 0.0
                # check for termination
                dual_val = prox_dual(ytrial, uk, alphak)
                primal_val_proj = prox_primal(
                    xtrial_proj, uk, alphak, self.func(xtrial_proj))
                gap = (primal_val_proj - dual_val)[0][0]
                if gap < self.targap:
                    xtrial = xtrial_proj
                    self.flag = 'desired'
                    break
                primal_val = prox_primal(
                    xtrial, uk, alphak, self.func(xtrial))
                gap = (primal_val - dual_val)[0][0]
                if gap < self.targap:
                    self.flag = 'desired'
                    break
                if self.inner_its > config['subsolver']['iteration_limits']:
                    self.flag = 'maxiters'
                    break
                y_current = ytrial
            self.gap = gap
            self.aoptim = l2_norm(xtrial - xk)
            self.xtrial = xtrial
            return xtrial, ytrial, self.aoptim

    def _proj_norm_ball(self, y):
        return _proj_norm_ball_jit(y, self.K, self.starts, self.ends, self.weights)

    def _get_group_structure(self, X):
        return self._get_group_structure_jit(X, self.K, self.starts, self.ends)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_group_structure_jit(X, K, starts, ends):
        nz = 0
        for i in range(K):
            start, end = starts[i], ends[i]
            X_Gi = X[start:end]
            if (np.sum(np.abs(X_Gi)) == 0):
                nz += 1
        nnz = K - nz
        return nnz, nz

    def print_header(self, config):
        # printlevel == 1
        if config['subsolver']['compute_exactpg']:
            header = "    aoptim/optim     its.  Flag  "
        else:
            header = "  aoptim   its.   Flag "
        if config['mainsolver']['print_level'] == 2:
            header += " Stepsize     Gap       TarGap "
            if config['subsolver']['compute_exactpg']:
                header += "     #az/#z   #anz/#nz |"
            else:
                header += "   #az  #anz |"
        else:
            header += "|"

        return header

    def print_iteration(self, config):
        # printlevel == 1
        if config['subsolver']['compute_exactpg']:
            content = f" {self.aoptim:.3e}/{self.optim:.3e} {self.inner_its:4d} {self.flag}"
        else:
            content = f" {self.aoptim:.3e} {self.inner_its:4d} {self.flag}"
        if config['mainsolver']['print_level'] == 2:
            content += f" {self.stepsize:.3e} {self.gap:+.3e} {self.targap:+.3e}"
            nnz, nz = self._get_group_structure(self.xtrial)
            if config['subsolver']['compute_exactpg']:
                content += f" {nz:4d}/{self.zeroGroup:4d}  {nnz:4d}/{self.nonZeroGroup:4d} |"
            else:
                content += f" {nz:4d}  {nnz:4d} |"
        else:
            content += "|"
        return content
