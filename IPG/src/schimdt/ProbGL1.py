'''
File: ProbGL1.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 02:01
Last Modified: 2021-04-06 00:38
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

    def ipg(self, xk, gradfxk, alphak, epsilon, init_perturb, mode, seed, max_attempts):
        self.seed = seed
        xprox, self.nz, self.nnz = self._pg(xk, gradfxk, alphak)
        self.xprox = xprox
        perturb = init_perturb
        attempt = 1
        while True:
            if attempt > max_attempts:
                warnings.warn("_ipg_sample: cannot sample a (x,y) pair satisfying the gap condition!")
                return None
            x, y = self._sample_primal_dual(xprox, xk, gradfxk, alphak, perturb, mode)
            gap = self._duality_gap(x, y, xk, gradfxk, alphak)
            # print(f"gap:{gap:3.5e} | target:{epsilon:3.5e}")
            if gap <= epsilon:
                self.attempt = attempt
                self.perturb = perturb
                self.gap = gap
                # print("=======")
                return x
            perturb *= 0.8
            attempt += 1

    def _sample_primal_dual(self, xprox, xk, gradfxk, alphak, perturb, mode='whole', **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed)
        if mode == 'whole':
            delta = np.random.randn(*xprox.shape)
            delta_norm = utils.l2_norm(delta)
            delta *= (perturb / delta_norm)
            x = xprox + delta
        elif mode == 'blocks':
            raise utils.AlgorithmError(f"mode:{mode} is not implemented yet.")
        else:
            raise utils.AlgorithmError(f"mode:{mode} is not defined.")
        # print(f"per:{perturb} | diff:{utils.l2_norm(x-xprox)}")
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
