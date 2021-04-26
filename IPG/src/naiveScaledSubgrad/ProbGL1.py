'''
File: ProbGL1.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 02:01
Last Modified: 2021-04-18 17:23
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
np.seterr(over='raise')


class ProbGL1(Problem):
    def __init__(self, f, r) -> None:
        super().__init__(f, r)
        self.K = self.r.K
        self.n, self.p = self.f.n, self.f.p
        self.starts = self.r.starts
        self.ends = self.r.ends
        self.Lambda_group = self.r.Lambda_group
        self.overflow = False

    def funcf(self, x):
        return self.f.evaluate_function_value(x)

    def funcr(self, x):
        return self.r.evaluate_function_value_jit(x)

    def gradf(self, x):
        return self.f.gradient()

    def ipg(self, xk, gradfxk, alphak, method, **kwargs):
        # place holder in case the _pg method is not available
        self.nz, self.nnz = None, None
        xprox, self.nz, self.nnz = self._pg(xk, gradfxk, alphak)
        self.xprox = xprox
        epsilon_safeguard = kwargs['epsilon_safeguard']
        t = kwargs['t']
        if method == 'sampling':
            init_perturb = kwargs['init_perturb']
            mode = kwargs['mode']
            seed = kwargs['seed']
            max_attempts = kwargs['max_attempts']
            x = self._ipg_sample(xprox, xk, gradfxk, alphak, init_perturb, epsilon_safeguard, max_attempts, t, mode, seed)
        elif method == 'subgradient':
            x_init = kwargs['x_init']
            x = self._ipg_subgradient(xk, gradfxk, alphak, x_init, t, epsilon_safeguard, maxiter=kwargs['maxiter_inner'])
        else:
            raise ValueError(f'{method} is not defined.')
        # print(f"probGL1: x:{x.T}")
        # print(f"xaprox-xprox:{utils.linf_norm(x-xprox)}")
        return x

    def _ipg_subgradient(self, xk, gradfxk, alphak, x_init, t, epsilon_safeguard, maxiter):
        if x_init is None:
            x = np.zeros_like(xk)
        else:
            x = x_init
        iters = 0
        self.ck = (np.sqrt(6 / ((1 + t) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
        # fbest = np.inf
        # # ref
        # # gap reference
        # ref = xk - alphak * gradfxk
        # temp = (self.xprox - ref) / alphak
        # dual_norm = self.r.dual(temp)
        # yref = min(1, 1 / dual_norm) * temp
        # # check duality gap
        # gapref = self._duality_gap(self.xprox, yref, xk, gradfxk, alphak)
        # print(f'gapref:{gapref:3.3e}')
        while True:
            # generate dual variable
            gradient_step = xk - alphak * gradfxk
            temp = (x - gradient_step) / alphak
            dual_norm = self.r.dual(temp)
            y = min(1, 1 / dual_norm) * temp
            # check duality gap
            gap = self._duality_gap(x, y, xk, gradfxk, alphak)
            if self.overflow:
                warnings.warn("_ipg_subgradient: too large subgradient leads to overflow!")
                return None
            # check termination
            diff = x - xk
            myepsilon = self.ck * (np.dot(diff.T, diff)[0][0])
            if myepsilon <= epsilon_safeguard:
                criterion = myepsilon
                # adaptive epsilon
                eflag = 'a'
            else:
                criterion = epsilon_safeguard
                # safeguared epsilon
                eflag = 's'
            # fcurrent = self._prox_primal(x, xk, gradfxk, alphak)
            # if fcurrent < fbest:
            #     fbest = fcurrent
            self.gap = gap
            self.epsilon = criterion
            self.attempt = iters
            if gap <= criterion:
                # distance to the exact proximal point
                self.perturb = utils.l2_norm(x - self.xprox)
                self.eflag = eflag
                return x
            if iters > maxiter:
                warnings.warn("_ipg_subgradient: cannot find a (x,y) pair satisfying the gap condition!")
                return None
            subgrad = _get_subgradient_prox_prob(x, xk, alphak, gradfxk,
                                                 self.K, self.starts,
                                                 self.ends, self.Lambda_group)
            # gradcheat = temp + self.r.evaluate_gradient_jit(x)
            # print(utils.l2_norm(gradcheat - subgrad))
            norm_subgrad = utils.l2_norm(subgrad)
            if norm_subgrad <= 1e-16:
                warnings.warn("_ipg_subgradient: norm of subgradient is zero! But do not satisfy the relative termination condition!")
                return None
            # print(f"iters:{iters:5d} | gap:{gap:3.3e} | epsilon:{criterion:3.3e} | which:{eflag} | fcurrent:{fcurrent:3.6e} | fbest:{fbest:3.6e} | subgrad:{norm_subgrad:3.3e}")
            # print(f"iters:{iters:5d} | gap:{gap:3.3e} | epsilon:{criterion:3.3e} | prox-aprox:{utils.l2_norm(x-self.xprox):3.3e}| subgrad:{norm_subgrad:3.3e}")
            # print('iter:\n', '   ', x.T, '\n', '    ', flags)
            # # print('subgrad:\n', '   ', subgrad.T)
            # print("==============================")
            # go to the next iteration
            iters += 1
            # stepsize = 1 / iters
            # stepsize = 2 * alphak / (iters + 1)
            stepsize = 1 / (iters + 1)
            # x = x - stepsize * subgrad
            x = x - (stepsize / norm_subgrad) * subgrad

    def _prox_primal(self, x, xk, gradfxk, alphak):
        gradient_step = xk - alphak * gradfxk
        temp = x - gradient_step
        return np.dot(temp.T, temp)[0][0] / (2 * alphak) + self.r.evaluate_function_value_jit(x)

    def _ipg_sample(self, xprox, xk, gradfxk, alphak, init_perturb, epsilon_safeguard, max_attempts, t, mode, seed):
        self.seed = seed
        # print(self.seed)
        perturb = init_perturb
        self.ck = (np.sqrt(6 / ((1 + t) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
        attempt = 1
        while True:
            if attempt > max_attempts:
                warnings.warn("_ipg_sample: cannot sample a (x,y) pair satisfying the gap condition!")
                return None
            x, y = self._sample_primal_dual(xprox, xk, gradfxk, alphak, perturb, mode)
            diff = x - xk
            gap = self._duality_gap(x, y, xk, gradfxk, alphak)
            # print(f"gap:{gap:3.5e} | target:{self.ck * (np.dot(diff.T, diff)[0][0]):3.5e} | diff_norm:{np.dot(diff.T, diff)[0][0]:3.5e} | ck:{self.ck:3.5e}")
            myepsilon = self.ck * (np.dot(diff.T, diff)[0][0])
            if myepsilon <= epsilon_safeguard:
                criterion = myepsilon
                # adaptive epsilon
                eflag = 'a'
            else:
                criterion = epsilon_safeguard
                # safeguared epsilon
                eflag = 's'
            if gap <= criterion:
                self.attempt = attempt
                # distance to the exact proximal point
                self.perturb = perturb
                self.gap = gap
                self.eflag = eflag
                self.epsilon = criterion
                # print(f"per:{perturb} | attempt:{attempt} | con:{0.8**(attempt-1)*init_perturb} ")
                # print('==')
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
        try:
            primal = np.dot(temp.T, temp)[0][0] / (2 * alphak) + self.r.evaluate_function_value_jit(x)
        except Exception as e:
            print(e)
            primal = 0
            self.overflow = True
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


@jit(nopython=True, cache=True)
def _get_subgradient_prox_prob(x, xk, alphak, gradfxk, K, starts, ends, Lambda_group):
    # calculate the subgradient at the x
    diff = x - (xk - alphak * gradfxk)
    subgradient = np.zeros_like(diff)
    # flags = ['ss'] * K
    for i in range(K):
        start, end = starts[i], ends[i]
        xGi = x[start:end]
        norm_xG1 = np.sqrt(np.dot(xGi.T, xGi))[0][0]
        if norm_xG1 <= 1e-16:
            grad_smooth_part_scaled = diff[start:end] / (Lambda_group[i] * alphak)
            norm_grad_smooth_part_scaled = np.sqrt(np.dot(grad_smooth_part_scaled.T, grad_smooth_part_scaled))[0][0]
            if norm_grad_smooth_part_scaled <= 1:
                subgrad_regularizer = -grad_smooth_part_scaled
            else:
                subgrad_regularizer = -grad_smooth_part_scaled / norm_grad_smooth_part_scaled
            subgradient[start:end] = diff[start:end] / alphak + Lambda_group[i] * subgrad_regularizer
            # d = np.zeros_like(xGi)
            # d[0] = 1
            # subgradient[start:end] = diff[start:end] / alphak + Lambda_group[i] * d
            # flags[i] = 'non-smooth'
        else:
            subgradient[start:end] = diff[start:end] / alphak + (Lambda_group[i] / norm_xG1) * xGi
            # flags[i] = 'smooth'
    return subgradient
