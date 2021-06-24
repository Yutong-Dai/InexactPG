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
import src.natOG.printUtils as printUtils
import os

class Solver:
    def __init__(self, prob, params):
        self.prob = prob
        self.__dict__.update(params)
        self.params = params
        self.fall_back = False
        self.first = True
        self.version = "0.2 (2021-06-18) natOG"
    def _set_init_alpha(self, x):
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
    def _linesearch(self, x, alpha, fvalx, rvalx, gradfx, flag):
        self.do_linesearch = False
        xtrial = x + self.d
        fval_xtrial = self.prob.funcf(xtrial)
        rval_xtrial = self.prob.funcr(xtrial)
        self.stepsize = 1
        self.bak = 0
        if self.params['inexact_type'] == 1:
            dirder_upper = -(self.d_norm_sq / alpha - np.sqrt(2 * self.epsilon / alpha) * self.d_norm - self.epsilon)
            # print(f'part1:{self.d_norm_sq / alpha:3.3e} | part2:{np.sqrt(2 * self.epsilon / alpha) * self.d_norm:3.3e} | part3:{self.epsilon:3.3e} | dirder_upper:{dirder_upper:3.3e} | flag:{flag}')
            # print(f'dirder_upper:{dirder_upper:3.3e} | flag:{flag}')
            if dirder_upper > 0:
                self.fall_back = True
                # self.update_alpha_strategy = 'frac'
                return xtrial, fval_xtrial, rval_xtrial
            const = self.eta *dirder_upper
        elif self.params['inexact_type'] == 2:
            dirder_upper = -(rvalx - rval_xtrial - np.sum(gradfx * self.d))
            if dirder_upper > 0:
                self.fall_back = True
                # self.update_alpha_strategy = 'frac'
                return xtrial, fval_xtrial, rval_xtrial
            const = self.eta * dirder_upper
        else:
            return xtrial, fval_xtrial, rval_xtrial
        while True:
            lhs = fval_xtrial - fvalx + rval_xtrial - rvalx
            rhs = self.stepsize * const
            # print(f"lhs:{lhs:3.3e} | rhs:{rhs:3.3e}")
            if lhs <= rhs:
                self.do_linesearch = True
                # print("========")
                return xtrial, fval_xtrial, rval_xtrial
            if self.bak == self.max_back:
                # line search failed
                self.do_linesearch = True
                print(f"lhs:{fval_xtrial - fvalx + rval_xtrial - rvalx:3.3e} | rhs:{self.stepsize * const:3.3e} | dirder_upper:{dirder_upper:3.3e}")
                if flag == "numetol_":
                    self.fall_back = True
                    # self.update_alpha_strategy = 'frac'
                    return xtrial, fval_xtrial, rval_xtrial
                self.status = -1
                return None, None, None
            self.bak += 1
            self.stepsize *= self.params['xi']
            xtrial = x + self.stepsize * self.d
            fval_xtrial = self.prob.funcf(xtrial)
            rval_xtrial = self.prob.funcr(xtrial)
        
    def solve(self, x=None, alpha=None, explore=True):
        if x is None:
            x = np.zeros((self.prob.p, 1))
        if not alpha:
            # alpha = self._set_init_alpha(x)
            alpha = 1.0
        if self.params['ckpt']:
            self.ckpt_dir = f"../log/{self.params['probSetAttr']['date']}/{self.params['inexact_type']}/{self.params['probSetAttr']['loss']}_ckpt/"
            self.ckpt_dir += f"{self.params['subsolver']}_{self.params['warm_start']}_{self.params['probSetAttr']['lambda_shrinkage']}"
            self.ckpt_dir += f"_{self.params['probSetAttr']['group_size']}_{self.params['probSetAttr']['overlap_ratio']}"
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
        rvalx = self.prob.funcr(x)
        Fvalx = fvalx + rvalx
        if explore:
            Fseq.append(Fvalx)
        gradfx = self.prob.gradf(x)
        fevals += 1
        gevals += 1
        Y = None
        subits = 0
        subfevals = 0
        subgevals = 0
        # count subsits in full dualDim computation
        subits_equiv = 0
        # consequtive_no_bak = 0
        while True:
            # if iteration == 40:
            #     print("iter 40!")
            print_start = time.time()
            if self.printlevel > 0:
                if iteration % self.printevery == 0:
                    printUtils.print_header(outID)
                printUtils.print_iterates(iteration, Fvalx, outID)
            print_cost = time.time() - print_start

            # ------------------- proximal operator calculation ---------------------------------------
            if Y is None or self.params['warm_start'] == False:
                Y_init = None
            else:
                Y_init = Y
            xaprox, Y, flag, subit, gap, epsilon, theta = self.prob.ipg(x, gradfx, alpha, self.params,
                                                                outter_iter=iteration + 1,
                                                                Y_init=Y_init, outID=outID)
            self.epsilon = epsilon
            # self.xaprox= xaprox
            if gap == -999:
                self.status = -2
            subits += subit
            subits_equiv += subit * self.prob.dualProbDim / len(Y) 
            subfevals += self.prob.fevals
            subgevals += self.prob.gevals
            nz = np.sum(xaprox == 0)
            nnz = self.prob.p - nz
            # collect stats
            # switch from 2-norm to inf-norm
            # self.aprox_optim = utils.l2_norm(self.aprox - x)
            self.d = xaprox - x
            aprox_optim = utils.linf_norm(self.d)
            self.d_norm = utils.l2_norm(self.d)
            self.d_norm_sq = self.d_norm ** 2
 

            iteration_cost = time.time() - iteration_start - print_cost
            time_so_far += iteration_cost
            if self.printlevel > 0 and flag != 'lnscfail':
                if explore:
                    Eseq.append(epsilon)
                    Gseq.append(gap)
                printUtils.print_proximal_update(alpha, self.prob.dualProbDim, subit, flag, gap,
                                                 epsilon, theta, aprox_optim, nz, nnz, outID)
            else:
                printUtils.print_proximal_update_failed(alpha, self.prob.dualProbDim, subit, flag, outID)

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
            if self.params['ckpt']:
                if aprox_optim <= self.params['ckpt_tol']:
                    info = {'X': x, 'iteration': iteration, 'time': time_so_far, 'F': Fvalx,
                            'nz': nz, 'nnz': nnz, 'status': self.status,
                            'fevals': fevals, 'gevals': gevals, 'optim': aprox_optim,
                            'n': self.prob.n, 'p': self.prob.p, 'Lambda': self.prob.r.Lambda,
                            'K': self.prob.K, 'subits': subits, 'subits_equiv':subits_equiv,
                            'subfevals':subfevals, 'subgevals':subgevals}
                    info['datasetid'] = self.datasetid
                    info_name = self.ckpt_dir + "/{}_info.npy".format(self.datasetname_)
                    np.save(info_name, info)

            iteration_start = time.time()
            # linesearch
            xtrial, fval_xtrial, rval_xtrial = self._linesearch(x, alpha, fvalx, rvalx, gradfx, flag)
            if self.status == -1:
                break
            fevals += (self.bak + 1)
            temp = time.time()
            if self.printlevel > 0:
                printUtils.print_linesearch(self.d_norm, self.bak, self.stepsize, outID)
            print_cost = time.time() - temp
            # update proximal gradient stepsize
            if self.params['inexact_type'] != 3:
                if self.do_linesearch:
                    if not self.fall_back:
                        if self.update_alpha_strategy == 'frac':
                            if self.bak > 0:
                                alpha *= self.zeta
                        elif self.update_alpha_strategy == 'model':
                            if flag == 'desired_':
                                d = xtrial - x
                                d_norm_sq = utils.l2_norm(d) ** 2
                                dirder = np.dot(gradfx.T, d)[0][0]
                                actual_decrease = fval_xtrial - fvalx
                                L_local = 2 * (actual_decrease - dirder) / d_norm_sq
                                alpha = min(max(1 / max(L_local, 1e-3), alpha * self.zeta), 10)
                                # print(f'iteration:{iteration:3d} | actual_decrease:{+actual_decrease:3.3e} | dirder:{+dirder:3.3e} | d_norm_sq:{d_norm_sq:3.3e} | L_local:{L_local:3.3e} | alpha:{alpha:3.3e}')
                                
                        elif self.update_alpha_strategy == 'none':
                            pass
                        else:
                            raise ValueError(f'Invalid update_alpha_strategy: {self.update_alpha_strategy}')
                    else:
                        if self.first:
                            alpha = 1.0
                            self.first = False
                        if self.bak > 0:
                            alpha *= self.zeta
                else:
                    if self.fall_back:
                        if self.first:
                            alpha = 1.0
                            self.first = False
                        if self.bak > 0:
                            alpha *= self.zeta
                iteration += 1
                x = xtrial
                fvalx = fval_xtrial
                rvalx = rval_xtrial
                Fvalx = fvalx + rvalx
                gradfx = self.prob.gradf(x)
                gevals += 1
            else:
                # test L
                opt_3_stay = False
                diff = xtrial - x
                if fval_xtrial - fvalx > np.sum(gradfx * diff) + 1/(2*alpha) * np.sum(diff * diff):
                    opt_3_stay = True
                    alpha *= self.zeta
                if opt_3_stay:
                    iteration += 1
                    x = x
                    gradfx = self.prob.gradf(x)
                    gevals += 1
                else:
                    # perform update
                    iteration += 1
                    x = xtrial
                    fvalx = fval_xtrial
                    rvalx = rval_xtrial
                    Fvalx = fvalx + rvalx
                    gradfx = self.prob.gradf(x)
                    gevals += 1
            if explore:
                Fseq.append(Fvalx)

        info = {
            'X': x, 'iteration': iteration, 'time': time_so_far, 'F': Fvalx,
            'nz': nz, 'nnz': nnz, 'status': self.status,
            'fevals': fevals, 'gevals': gevals, 'optim': aprox_optim,
            'n': self.prob.n, 'p': self.prob.p, 'Lambda': self.prob.r.Lambda,
            'K': self.prob.K, 'subits': subits, 'subits_equiv':subits_equiv,
            'subfevals':subfevals, 'subgevals':subgevals}
        if explore:
            info['Fseq'] = Fseq
            info['Eseq'] = Eseq
            info['Gseq'] = Gseq
        if self.printlevel > 0:
            printUtils.print_exit(self.status, outID)
            printUtils.print_result(info, outID)
        return info
