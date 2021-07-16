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
import os

class Solver:
    def __init__(self, prob, params):
        self.prob = prob
        self.__dict__.update(params)
        self.params = params
        self.version = "0.2 (2021-06-28) latentOG"

    def solve(self, x=None, alpha=None, explore=True, provide_L=False):
        if x is None:
            x = np.zeros((self.prob.p, 1))
        if not alpha:
            alpha = 2.0

        if self.params['scale_alpha']:
            alpha = (1 - self.params['eta']) * alpha
            if self.params['inexact_type'] == 1:
                self.params['gamma1'] = 0.5
        else:
            if provide_L:
                if self.params['inexact_type'] == 1:
                    self.params['gamma1'] = 1 / (2 - 3 * self.params['eta'])
                    # such that alpha = 1/L
                if self.params['inexact_type'] == 2:
                    alpha = (1 - self.params['eta']) * alpha
        if self.params['ckpt']:
            self.ckpt_dir = f"../log/{self.params['probSetAttr']['date']}/{self.params['inexact_type']}/{self.params['probSetAttr']['loss']}_ckpt/"
            self.ckpt_dir += f"{self.params['subsolver']}_{self.params['warm_start']}_{self.params['probSetAttr']['lambda_shrinkage']}"
            self.ckpt_dir += f"_{self.params['probSetAttr']['group_size']}_{self.params['probSetAttr']['overlap_ratio']}"
            self.ckpt_dir += f"_{self.params['probSetAttr']['param1']}_{self.params['probSetAttr']['param2']}"
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            self.datasetname_ = self.prob.f.datasetName.split("/")[-1]
            self.datasetid = "{}_{}_{}_{}".format(self.datasetname_, self.params['probSetAttr']['lambda_shrinkage'], 
                                                               self.params['probSetAttr']['group_size'], 
                                                               self.params['probSetAttr']['overlap_ratio'])
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
        subfevals = 0
        subgevals = 0
        # count subsits in full dualDim computation
        subits_equiv = 0
        while True:
            print_start = time.time()
            if self.printlevel > 0:
                if iteration % self.printevery == 0:
                    printUtils.print_header_lee(outID)
                printUtils.print_iterates(iteration, fvalx, outID)
            print_cost = time.time() - print_start

            # ------------------- proximal operator calculation ---------------------------------------
            if lambda_full is None or self.params['warm_start'] == False:
                lambda_init = None
            else:
                lambda_init = lambda_full
            self.bak = 0
            alpha_old = alpha
            subit_per_iter = 0
            while True:
                xaprox, lambda_full, flag, subit, gap, epsilon, theta, correction = self.prob.ipg(x, gradfx, alpha, self.params,
                                                                                                outter_iter=iteration + 1,
                                                                                                lambda_init=lambda_init, outID=outID)
                lambda_init = lambda_full                                                                                          
                subits += subit
                subit_per_iter += subit
                subits_equiv += subit * self.prob.dualProbDim / len(lambda_full) 
                subfevals += self.prob.fevals
                subgevals += self.prob.gevals
                if gap >= 1e10:
                    self.status = -2
                    break
                xtrial = xaprox
                diff = xtrial - x
                fval_xtrial = self.prob.funcf(xtrial)
                if fval_xtrial - fvalx > np.sum(gradfx * diff) + 1/(2*alpha) * np.sum(diff * diff):
                    alpha *= self.zeta
                    self.bak += 1
                    gradfx = self.prob.gradf(x)
                else:
                    break
            

            self.d = xaprox - x
            aprox_optim = utils.linf_norm(self.d)
            self.d_norm = utils.l2_norm(self.d)
            self.d_norm_sq = self.d_norm ** 2
            self.epsilon = epsilon
            self.subsolver_failed = False

            nz = np.sum(xaprox == 0)
            nnz = self.prob.p - nz

            iteration_cost = time.time() - iteration_start - print_cost
            time_so_far += iteration_cost
            if self.printlevel > 0 and flag != 'lnscfail':
                if explore:
                    Eseq.append(epsilon)
                    Gseq.append(gap)
                printUtils.print_proximal_update_lee(alpha_old, self.prob.dualProbDim, subit_per_iter, flag, gap,
                                                 epsilon, theta, correction, aprox_optim, nz, nnz, outID)
            else:
                printUtils.print_proximal_update(alpha_old, self.prob.dualProbDim, subit_per_iter, flag, outID)

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
            # # fake linsearch
            # xtrial = xaprox
            # fval_xtrial = self.prob.funcf(xtrial)
            temp = time.time()
            if self.printlevel > 0:
                printUtils.print_linesearch_lee(self.bak, outID)
            print_cost = time.time() - temp
            
            # perform update
            iteration += 1
            x = xtrial
            fvalx = fval_xtrial
            gradfx = self.prob.gradf(x)
            fevals += (self.bak + 1)
            gevals += (self.bak + 1)

            if self.params['ckpt']:
                if aprox_optim <= self.params['ckpt_tol'] or iteration == 1:
                    info = {'X': x, 'iteration': iteration, 'time': time_so_far, 'F': fvalx,
                            'nz': nz, 'nnz': nnz, 'status': 0,
                            'fevals': fevals, 'gevals': gevals, 'optim': aprox_optim,
                            'n': self.prob.n, 'p': self.prob.p, 'Lambda': self.prob.r.Lambda,
                            'K': self.prob.K, 'subits': subits, 'subits_equiv':subits_equiv,
                            'subfevals':subfevals, 'subgevals':subgevals}
                    if iteration == 1:
                        info['X'] = None
                        info['status'] = 2
                    info['datasetid'] = self.datasetid
                    info_name = self.ckpt_dir + "/{}_info.npy".format(self.datasetname_)
                    np.save(info_name, info)

        info = {
            'X': x, 'iteration': iteration, 'time': time_so_far, 'F': fvalx,
            'nz': nz, 'nnz': nnz, 'status': self.status,
            'fevals': fevals, 'gevals': gevals, 'optim': aprox_optim,
            'n': self.prob.n, 'p': self.prob.p, 'Lambda': self.prob.r.Lambda,
            'K': self.prob.K, 'subits': subits, 'subits_equiv':subits_equiv,
            'subfevals':subfevals, 'subgevals':subgevals}
        if explore:
            # info['Fseq'] = Fseq
            info['Eseq'] = Eseq
            info['Gseq'] = Gseq
        if self.printlevel > 0:
            printUtils.print_exit(self.status, outID)
            printUtils.print_result(info, outID)
        return info
