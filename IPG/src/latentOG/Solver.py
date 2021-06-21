'''
File: Solver.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:44
Last Modified: 2021-06-12 10:32
--------------------------------------------
Description:
'''
import sys
sys.path.append("../")
import numpy as np
import time
import src.utils as utils
import src.latentOG.printUtils as printUtils


class Solver:
    def __init__(self, prob, params):
        self.prob = prob
        self.__dict__.update(params)
        self.params = params
        self.version = "0.2 (2021-06-12) latentOG"

    def solve(self, x=None, alpha=None, explore=True):
        if x is None:
            x = np.zeros((self.prob.p, 1))
        if not alpha:
            alpha = self.set_init_alpha(x)
        if self.params['scale_alpha']:
            alpha = (1 - self.params['eta']) * alpha
            if self.params['inexact_type'] == 1:
                self.params['gamma1'] = 0.5
        else:
            if self.params['inexact_type'] == 1:
                # self.params['gamma1'] = 1 / (2 - 3 * self.params['eta'])
                self.params['gamma1'] = 1 / (1 - 2 * self.params['eta'])
                alpha *= 3 / 2
            if self.params['inexact_type'] == 2:
                alpha = (1 - self.params['eta']) * alpha

        # print algorithm params
        outID = self.prob.f.datasetName
        problem_attribute = self.prob.f.__str__()
        problem_attribute += "Regularizer:{:.>44}\n".format(self.prob.r.__str__())
        problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format('', self.prob.r.Lambda)
        problem_attribute += "Number of groups:{:.>32}\n".format(self.prob.r.K)
        if self.printlevel > 0:
            printUtils.print_problem(problem_attribute, self.version, outID)
            # printUtils.print_algorithm(self.__dict__, outID)
            printUtils.print_algorithm(self.params, outID)
        # set up loggings
        info = {}
        iteration = 0
        fevals = 0
        gevals = 0
        self.status = None
        time_so_far = 0
        if explore:
            Fseq = []
            Eseq = []
            Gseq = []
        iteration_start = time.time()
        fvalx = self.prob.funcf(x)
        gradfx = self.prob.gradf(x)
        fevals += 1
        gevals += 1
        lambda_full = None
        subits = 0
        # count subsits in full dualDim computation
        subits_equiv = 0
        while True:
            print_start = time.time()
            if self.printlevel > 0:
                if iteration % self.printevery == 0:
                    printUtils.print_header(outID)
                printUtils.print_iterates(iteration, fvalx, outID)
            print_cost = time.time() - print_start

            # ------------------- proximal operator calculation ---------------------------------------
            if lambda_full is None or self.params['warm_start'] == False:
                lambda_init = None
            else:
                lambda_init = lambda_full
            xaprox, lambda_full, flag, subit, gap, epsilon, theta, correction = self.prob.ipg(x, gradfx, alpha, self.params,
                                                                                              outter_iter=iteration + 1,
                                                                                              lambda_init=lambda_init, outID=outID)
            if gap >= 1e10:
                self.status = -2
            subits += subit
            subits_equiv += subit * self.prob.dualProbDim / len(lambda_full) 
            nz = np.sum(xaprox == 0)
            nnz = self.prob.p - nz
            # collect stats
            # switch from 2-norm to inf-norm
            # self.aprox_optim = utils.l2_norm(self.aprox - x)
            self.d = xaprox - x
            aprox_optim = utils.linf_norm(self.d)
            self.d_norm = utils.l2_norm(self.d)
            self.d_norm_sq = self.d_norm ** 2
            self.epsilon = epsilon
            self.subsolver_failed = False

            iteration_cost = time.time() - iteration_start - print_cost
            time_so_far += iteration_cost
            if self.printlevel > 0 and flag != 'lnscfail':
                if explore:
                    Eseq.append(epsilon)
                    Gseq.append(gap)
                printUtils.print_proximal_update(alpha, self.prob.dualProbDim, subit, flag, gap,
                                                 epsilon, theta, correction, aprox_optim, nz, nnz, outID)
            else:
                printUtils.print_proximal_update(alpha, self.prob.dualProbDim, subit, flag, outID)

            if iteration == 0:
                tol = max(1, aprox_optim) * self.tol
            if aprox_optim <= tol:
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
            # fake linsearch
            xtrial = xaprox
            fval_xtrial = self.prob.funcf(xtrial)
            temp = time.time()
            # if self.printlevel > 0:
            #     printUtils.print_linesearch(self.d_norm, 0, 1.0, outID)
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
            gradfx = self.prob.gradf(x)
            fevals += 1
            gevals += 1
            # boost numerical performance if beta > 1
            alpha *= self.beta
            # if iteration % 100 == 0:
            #     alpha *= 0.5

        info = {
            'X': x, 'iteration': iteration, 'time': time_so_far, 'f': fvalx,
            'nz': nz, 'nnz': nnz, 'status': self.status,
            'fevals': fevals, 'gevals': gevals, 'optim': aprox_optim,
            'n': self.prob.n, 'p': self.prob.p, 'Lambda': self.prob.r.Lambda,
            'K': self.prob.K, 'subits': subits, 'subits_equiv':subits_equiv}
        if explore:
            # info['Fseq'] = Fseq
            info['Eseq'] = Eseq
            info['Gseq'] = Gseq
        if self.printlevel > 0:
            printUtils.print_exit(self.status, outID)
            printUtils.print_result(info, outID)
        return info
