'''
File: ProbGL1.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 02:01
Last Modified: 2021-03-23 16:49
--------------------------------------------
Description:
'''
import sys
import os
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Problem import Problem
import utils as utils
import numpy as np
from numba import jit
import warnings


class ProbGL1(Problem):
    def __init__(self, f, r) -> None:
        super().__init__(f, r)
        self.K = self.r.K
        self.n, self.p = self.f.n, self.f.p
        self.starts = self.r.starts
        self.ends = self.r.ends
        self.Lambda_group = self.r.Lambda_group

    def funcf(self, x):
        return self.f.evaluate_function_value(x)

    def funcr(self, x):
        return self.r.evaluate_function_value_jit(x)

    def gradf(self, x):
        return self.f.gradient()

    def ipg(self, xk, gradfxk, alphak, method, **kwargs):
        # place holder in case the _pg method is not available
        self.nz, self.nnz = None, None
        if method == 'sampling':
            init_perturb = kwargs['init_perturb']
            t = kwargs['t']
            mode = kwargs['mode']
            seed = kwargs['seed']
            max_attempts = kwargs['max_attempts']
            xprox, self.nz, self.nnz = self._pg(xk, gradfxk, alphak)
            self.xprox = xprox
            x = self._ipg_sample(xprox, xk, gradfxk, alphak, init_perturb, max_attempts, t, mode, seed)
        elif method == 'algorithm':
            raise ValueError(f'{method} is not implemented.')
        else:
            raise ValueError(f'{method} is not defined.')
        return x

    def _ipg_sample(self, xprox, xk, gradfxk, alphak, init_perturb, max_attempts, t, mode, seed):
        self.seed = seed
        peturb = init_perturb
        self.ck = (np.sqrt(6 / ((1 + t) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
        attempt = 1
        while True:
            if attempt > max_attempts:
                warnings.warn("_ipg_sample: cannot sample a (x,y) pair satisfying the gap condition!")
                return None
            x, y = self._sample_primal_dual(xprox, xk, gradfxk, alphak, peturb, mode)
            diff = x - xk
            gap = self._duality_gap(x, y, xk, gradfxk, alphak)
            # print(f"gap:{gap} | target:{np.dot(diff.T, diff)[0][0]} | ck:{ck}")
            if gap <= self.ck * (np.dot(diff.T, diff)[0][0]):
                self.attempt = attempt
                # distance to the exact proximal point
                self.peturb = peturb
                self.gap = gap
                return x
            peturb *= 0.8
            attempt += 1

    def _sample_primal_dual(self, xprox, xk, gradfxk, alphak, peturb, mode='whole', **kwargs):
        if self.seed:
            np.random.seed(self.seed)
        if mode == 'whole':
            delta = np.random.randn(*xprox.shape)
            delta_norm = utils.l2_norm(delta)
            delta *= (peturb / delta_norm)
            x = xprox + delta
        elif mode == 'blocks':
            raise utils.AlgorithmError(f"mode:{mode} is not implemented yet.")
        else:
            raise utils.AlgorithmError(f"mode:{mode} is not defined.")
        gradient_step = xk - alphak * gradfxk
        temp = (x - gradient_step) / alphak
        dual_norm = self.r.dual(temp)
        y = min(1, 1 / dual_norm) * temp
        return x, y

    def _duality_gap(self, x, y, xk, gradfxk, alphak):
        gradient_step = xk - alphak * gradfxk
        temp = x - gradient_step
        primal = np.dot(temp.T, temp)[0][0] / (2 * alphak) + self.r.evaluate_function_value_jit(x)
        dual_negative = ((alphak / 2) * (np.dot(y.T, y)) + np.dot(gradient_step.T, y))[0][0]
        return primal + dual_negative

    def _pg(self, xk, gradfxk, alphak):
        xprox, zeroGroup, nonZeroGroup = _proximal_gradient_jit(xk, alphak, gradfxk,
                                                                self.K, self.p, self.starts,
                                                                self.ends, self.Lambda_group)
        return xprox, zeroGroup, nonZeroGroup


@jit(nopython=True, cache=True)
def _proximal_gradient_jit(X, alpha, gradf, K, p, starts, ends, Lambda_group):
    proximal = np.zeros((p, 1))
    nonZeroGroup = []
    zeroGroup = []
    for i in range(K):
        start, end = starts[i], ends[i]
        XG_i = X[start:end]
        gradfG_i = gradf[start:end]
        gradient_step = XG_i - alpha * gradfG_i
        gradient_step_norm = np.sqrt(np.dot(gradient_step.T, gradient_step))[0][0]
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
