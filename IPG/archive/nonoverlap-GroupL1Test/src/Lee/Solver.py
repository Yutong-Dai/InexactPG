'''
File: Solver.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:44
Last Modified: 2021-04-18 16:56
--------------------------------------------
Description:
'''
import sys
import os

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import numpy as np
import time
import src.utils as utils
import src.Lee.printUtils as printUtils


class Solver:
    def __init__(self, prob, params):
        self.prob = prob
        self.__dict__.update(params)
        self.version = "0.1.1 (2021-04-05)"

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

    def proximal_update(self, x, gradfx, alpha, scalesubgrad, epsilon_safeguard=1e9):
        self.bak = 0
        self.ipg_nnz, self.ipg_nz = None, None
        # incase the subgradient solver failed at the very beginning
        self.prox_optim = 1e9
        if self.inexact_strategy == 'sampling':
            self.xaprox = self.prob.ipg(x, gradfx, alpha, self.inexact_strategy,
                                        init_perturb=self.init_perturb, epsilon_safeguard=epsilon_safeguard,
                                        t=self.t, mode=self.mode, seed=0, max_attempts=self.max_attempts)
        else:
            self.xaprox = self.prob.ipg(x, gradfx, alpha, self.inexact_strategy,
                                        x_init=x, epsilon_safeguard=epsilon_safeguard,
                                        t=self.t, maxiter_inner=self.maxiter_inner,
                                        scalesubgrad=scalesubgrad, threshold=self.threshold)
        if self.xaprox is not None:
            # collect stats
            # switch from 2-norm to inf-norm
            # self.prox_optim = utils.l2_norm(self.prob.xprox - x)
            self.prox_optim = utils.linf_norm(self.prob.xprox - x)
            # self.aprox_optim = utils.l2_norm(self.xaprox - x)
            self.aprox_optim = utils.linf_norm(self.xaprox - x)
            # optional: get sparsity sturctures.
            if self.prob.nz is not None:
                self.pg_nz, self.pg_nnz = self.prob.nz, self.prob.nnz
            if self.pg_nz is not None:
                self.ipg_nnz = utils.get_group_structure(self.xaprox, self.prob.K, self.prob.starts, self.prob.ends)
                self.ipg_nz = self.prob.K - self.ipg_nnz
            self.d = self.xaprox - x
            self.d_norm = utils.l2_norm(self.d)
            self.d_norm_sq = self.d_norm ** 2
            self.epsilon = self.prob.epsilon
            self.subsolver_failed = False
        else:
            self.subsolver_failed = True
            # ipg solver failed
            self.epsilon = self.prob.epsilon
            self.status = -2

    def linesearch(self, x, alpha, fvalx, rvalx):
        # backtrack linesearch
        xtrial = x + self.d
        fval_xtrial = self.prob.funcf(xtrial)
        rval_xtrial = self.prob.funcr(xtrial)
        const = -self.eta * (self.d_norm_sq / alpha - np.sqrt(2 * self.epsilon / alpha) * self.d_norm - self.epsilon)
        # print(1 / alpha - np.sqrt(2 * self.prob.ck / alpha) - self.prob.ck)
        # print(f"LHS:{self.t/(2*alpha)*self.prox_optim**2: 3.6e} | RHS:{self.d_norm_sq / alpha - np.sqrt(2 * self.epsilon / alpha) * self.d_norm - self.epsilon}")
        self.stepsize = 1
        while True:
            # print(f"LHS:{fval_xtrial - fvalx + rval_xtrial - rvalx:3.3e} | RHS:{self.stepsize * const:3.3e} | stepsize:{self.stepsize:3.3e} | const:{const:3.3e}")
            if (fval_xtrial - fvalx + rval_xtrial - rvalx) <= self.stepsize * const:
                # print('===')
                return xtrial, fval_xtrial, rval_xtrial
            if self.bak == self.max_back:
                # line search failed
                self.status = -1
                return None, None, None
            self.bak += 1
            self.stepsize *= self.xi
            xtrial = x + self.stepsize * self.d
            fval_xtrial = self.prob.funcf(xtrial)
            rval_xtrial = self.prob.funcr(xtrial)

    def solve(self, x=None, alpha=None, explore=True, scalesubgrad=False):
        if x is None:
            x = np.zeros((self.prob.p, 1))
        if not alpha:
            alpha = self.set_init_alpha(x)

        # print algorithm params
        outID = self.prob.f.datasetName
        problem_attribute = self.prob.f.__str__()
        problem_attribute += "Regularizer:{:.>44}\n".format(self.prob.r.__str__())
        problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format('', self.prob.r.Lambda)
        problem_attribute += "Number of groups:{:.>32}\n".format(self.prob.r.K)
        if self.printlevel > 0:
            printUtils.print_problem(problem_attribute, self.version, outID)
            printUtils.print_algorithm(self.__dict__, outID)
        # set up loggings
        info = {}
        iteration = 0
        fevals = 0
        gevals = 0
        baks = 0
        self.status = None
        time_so_far = 0
        subgrad_iters = 0
        if explore:
            Fseq = []
            Eseq = []
            Gseq = []
        iteration_start = time.time()
        fvalx = self.prob.funcf(x)
        rvalx = self.prob.funcr(x)
        Fvalx = fvalx + rvalx
        if explore:
            Fseq.append(Fvalx)
        gradfx = self.prob.gradf(x)
        fevals += 1
        gevals += 1
        while True:
            print_start = time.time()
            if self.printlevel > 0:
                if iteration % self.printevery == 0:
                    printUtils.print_header(outID)
                printUtils.print_iterates(iteration, Fvalx, outID)
            print_cost = time.time() - print_start
            if self.safeguard_opt == 'const':
                epsilon_safeguard = self.safeguard_const
            elif self.safeguard_opt == 'schimdt':
                epsilon_safeguard = self.safeguard_const / (iteration + 1)**3
            elif self.safeguard_opt == 'laststep':
                if iteration == 0:
                    epsilon_safeguard = self.safeguard_const
                else:
                    # from last iteration
                    epsilon_safeguard = self.safeguard_const * (self.d_norm * self.stepsize) ** 2 * self.prob.ck
            elif self.safeguard_opt == 'none':
                epsilon_safeguard = np.inf
            self.proximal_update(x, gradfx, alpha, scalesubgrad, epsilon_safeguard)
            iteration_cost = time.time() - iteration_start - print_cost
            time_so_far += iteration_cost
            if self.printlevel > 0:
                if not self.subsolver_failed:
                    # change from 2-norm to inf-norm
                    # prox_diff = utils.l2_norm(self.xaprox - self.prob.xprox)
                    # print(f"outter: xaprox-xprox:{utils.linf_norm(self.xaprox - self.prob.xprox)}")
                    prox_diff = utils.linf_norm(self.xaprox - self.prob.xprox)
                    if explore:
                        Eseq.append(self.epsilon)
                        Gseq.append(self.prob.gap)
                    printUtils.print_proximal_update(alpha, self.t, self.prob.ck, self.prob.attempt, self.prob.gap,
                                                     self.epsilon, self.prob.eflag, prox_diff, self.prox_optim, self.aprox_optim,
                                                     self.pg_nnz, self.ipg_nnz, self.pg_nz, self.ipg_nz, outID)
                    subgrad_iters += self.prob.attempt
                else:
                    printUtils.print_proximal_update_failed(alpha, self.t, self.prob.ck, self.prob.attempt, self.prob.gap,
                                                            self.epsilon, outID)
            if iteration == 0:
                tol = max(1, self.prox_optim) * self.tol
            if self.prox_optim <= tol:
                self.status = 0
                break
            if iteration >= self.max_iter:
                self.status = 1
                break
            if time_so_far >= self.max_time:
                self.status = 2
                break
            if self.status == -2:
                break

            iteration_start = time.time()
            # linesearch
            xtrial, fval_xtrial, rval_xtrial = self.linesearch(x, alpha, fvalx, rvalx)
            fevals += (self.bak + 1)
            baks += self.bak
            temp = time.time()
            if self.printlevel > 0:
                printUtils.print_linesearch(self.d_norm, self.bak, self.stepsize, outID)
            print_cost = time.time() - temp
            # update proximal gradient stepsize
            if self.update_alpha_strategy == 'frac':
                if self.bak > 0:
                    alpha *= self.zeta
            elif self.update_alpha_strategy == 'model':
                d = xtrial - x
                d_norm_sq = utils.l2_norm(d) ** 2
                dirder = np.dot(gradfx.T, d)[0][0]
                actual_decrease = fval_xtrial - fvalx
                L_local = 2 * (actual_decrease - dirder) / d_norm_sq
                alpha = max(1 / max(L_local, 1e-8), alpha * self.zeta)
            elif self.update_alpha_strategy == 'none':
                pass
            else:
                raise ValueError(f'Invalid update_alpha_strategy: {self.update_alpha_strategy}')

            # perform update
            iteration += 1
            x = xtrial
            fvalx = fval_xtrial
            rvalx = rval_xtrial
            Fvalx = fvalx + rvalx
            gradfx = self.prob.gradf(x)
            gevals += 1
            # boost numerical performance if beta > 1
            alpha *= self.beta
            if explore:
                Fseq.append(Fvalx)

        info = {
            'X': x, 'iteration': iteration, 'time': time_so_far, 'F': Fvalx,
            'nz': self.ipg_nz, 'nnz': self.ipg_nnz, 'status': self.status,
            'fevals': fevals, 'gevals': gevals, 'baks': baks, 'optim': self.prox_optim,
            'n': self.prob.n, 'p': self.prob.p, 'Lambda': self.prob.r.Lambda,
            'K': self.prob.K, 'subgrad_iters': subgrad_iters
        }
        if explore:
            info['Fseq'] = Fseq
            info['Eseq'] = Eseq
            info['Gseq'] = Gseq
        if self.printlevel > 0:
            printUtils.print_exit(self.status, outID)
            printUtils.print_result(info, outID)
        return info
