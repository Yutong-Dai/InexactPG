'''
File: regularizer.py
Author: Yutong Dai
File Created: 2020-05-11 21:18
Last Modified: 2020-11-16 17:57
--------------------------------------------
Description:
'''
import numpy as np
from numba import jit
from numpy.linalg import norm


@jit(nopython=True)
def _f(X, K, starts, ends, Lambda_group):
    fval = 0.0
    for i in range(K):
        start, end = starts[i], ends[i]
        XG_i = X[start:end]
        fval += Lambda_group[i] * np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
    return fval


@jit(nopython=True)
def _dual_jit(y, K, starts, ends, Lambda_group):
    """
    compute the dual of r(x), which is r(y): max ||y_g||/lambda_g
    reference: https://jmlr.org/papers/volume18/16-577/16-577.pdf section 5.2
    """
    max_group_norm = 0.0
    for i in range(K):
        start, end = starts[i], ends[i]
        yG_i = y[start:end]
        temp_i = (np.sqrt(np.dot(yG_i.T, yG_i))[0][0]) / Lambda_group[i]
        max_group_norm = max(max_group_norm, temp_i)
    return max_group_norm


@jit(nopython=True, cache=True)
def _compute_proximal_gradient_update_jit(X, alpha, gradf, starts, ends, Lambda_group):
    proximal = np.zeros_like((X, 1))
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
            temp = 1 - ((Lambda_group[i] * alpha) / gradient_step_norm)
        else:
            temp = -1
        if temp > 0:
            nonZeroGroup.append(i)
        else:
            zeroGroup.append(i)
        proximal[start:end] = max(temp, 0) * gradient_step
    return proximal, len(zeroGroup), len(nonZeroGroup)


class GL1:
    def __init__(self, group, Lambda=None, Lambda_group=None):
        """
        !!Warning: need `group` be ordered in a consecutive manner, i.e.,
        group: array([1., 1., 1., 2., 2., 2., 3., 3., 3., 3.])
        Then:
        unique_groups: array([1., 2., 3.])
        group_frequency: array([3, 3, 4]))
        """
        self.group = group
        self.unique_groups, self.group_frequency = np.unique(
            self.group, return_counts=True)
        if Lambda_group is not None:
            self.Lambda_group = Lambda_group
        else:
            if Lambda is not None:
                self.Lambda_group = Lambda * np.sqrt(self.group_frequency)
            else:
                raise ValueError("Initialization failed!")
        self.K = len(self.unique_groups)
        self.group_size = -1 * np.ones(self.K)
        p = group.shape[0]
        full_index = np.arange(p)
        starts = []
        ends = []
        for i in range(self.K):
            G_i = full_index[np.where(self.group == self.unique_groups[i])]
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
        return _f(X, self.K, self.starts, self.ends, self.Lambda_group)

    # gradient is calculated on the fly, no need to define a method here.

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
            hv[start:end] = self.Lambda_group[i] * (1 / self.hv_data[i]['XG_i_norm'] * vi -
                                                    (temp / self.hv_data[i]['XG_i_norm_cubic']) *
                                                    self.hv_data[i]['XG_i'])
        return hv

    def _dual(self, y):
        """
            compute the dual norm.
        """
        return _dual_jit(y, self.K, self.starts, self.ends, self.Lambda_group)

    def compute_proximal_gradient_update(self, xk, alphak, gradfxk, jit=True):
        if jit:
            prox, zeroGroup, nonZeroGroup = _compute_proximal_gradient_update_jit(xk, alphak, gradfxk, self.starts,
                                                                                  self.ends, self.Lambda_group)
        else:
            nonZeroGroup = 0
            zeroGroup = 0
            prox = np.zeros_like(xk)
            uk = xk - alphak * gradfxk
            for i in range(len(self.starts)):
                start, end = self.starts[i], self.ends[i]
                gradstep_Gi = uk[start:end]
                gradient_step_norm = norm(gradstep_Gi)
                if gradient_step_norm != 0:
                    temp = 1 - (self.Lambda_group[i]
                                * alphak / gradient_step_norm)
                else:
                    temp = -1
                if temp > 0:
                    nonZeroGroup += 1
                else:
                    zeroGroup += 1
            prox[start:end] = max(0, temp) * gradstep_Gi
        return prox, zeroGroup, nonZeroGroup

    def _projGL1Ball(self, y):
        projected_group = {}
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            y_Gi = y[start:end]
            norm_y_Gi = norm(y_Gi)
            if norm_y_Gi > self.Lambda_group[i]:
                y[start:end] = (self.Lambda_group[i] / norm_y_Gi) * y_Gi
                projected_group[i] = i
        return y, projected_group
