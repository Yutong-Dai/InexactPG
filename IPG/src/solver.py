'''
# File: solver.py
# Project: ipg
# Created Date: 2021-08-23 11:28
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-10-01 8:09
# Modified By: Yutong Dai yutongdai95@gmail.com
#
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
# 2021-08-23	Y.D	create file. Implement algorithm.
# 2021-09-17    Y.D Modify the Solver, which will record the best iterate encountered so far.
                    The best iterate is defined in terms of the newest iterate, where
                    the subsolver terminates successfully and  the trial iterate is accepted by the linesearch. 
                    If encounter 2 maxiters/correction steps, then terminate the algorithm
# 2021-09-29    Y.D Use the new termination  condition for the mainsolver, that is the upper-bound of the chi measure is 
#                   scaled by the alphak.                    
'''
import numpy as np
import datetime
import time
import os


class IpgSolver:
    def __init__(self, f, r, config):
        self.f = f
        self.r = r
        self.version = "0.1 (2021-09-17)"
        self.n = self.f.n
        self.p = self.f.p
        self.config = config
        self.datasetname = self.f.datasetName.split("/")[-1]
        if config['mainsolver']['log']:
            self.outID = self.f.datasetName
            self.filename = '{}.txt'.format(self.outID)
        else:
            self.filename = None
        if self.config['mainsolver']['print_level'] > 0:
            self.print_problem()
            self.print_config()

    def generate_ckpt_dir(self, save_ckpt_id, tol):
        ckpt_dir = f"./log/{save_ckpt_id['date']}"
        if self.config['mainsolver']['exact_pg_computation']:
            ckpt_dir += "/exact"
            additional_info = "_empty"
        else:
            name = self.config['mainsolver']['inexact_pg_computation']
            ckpt_dir += f"/{name}"
            if name == 'schimdt':
                additional_info = f"_{self.config['inexactpg'][name]['c']}"
            else:
                additional_info = f"_{self.config['inexactpg'][name]['gamma']}"
        ckpt_dir += f"/{save_ckpt_id['loss']}/ckpt/{tol}"
        ckpt_dir += f"/{save_ckpt_id['lam_shrink']}_{save_ckpt_id['grp_size']}_{save_ckpt_id['overlap_ratio']}" + additional_info
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        return ckpt_dir

    def solve(self, x_init=None, alpha_init=None, save_ckpt=False, save_ckpt_id=None, milestone=None):
        self.datasetid = None
        # process argument
        if save_ckpt:
            if save_ckpt_id is None:
                raise ValueError(
                    "`save_ckpt` is set to True but `save_ckpt_id` is None!")
            self.datasetid = f"{self.datasetname}_{save_ckpt_id['lam_shrink']}_{save_ckpt_id['grp_size']}_{save_ckpt_id['overlap_ratio']}"
            if milestone is None:
                raise ValueError(
                    "`save_ckpt` is set to True but `milestone` is None!")
            milestone = [*set(milestone)]
            milestone.append(-1.0)  # placeholder
            milestone.sort(reverse=True)
        # configure mainsolver
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init
        x_best_so_far = xk
        if not alpha_init:
            self.alphak = 1.0
        else:
            self.alphak = alpha_init
        tol = self.config['mainsolver']['accuracy']
        # dual variable
        yk = None
        # collect stats
        self.iteration = 0
        self.fevals = 0
        self.gevals = 0
        self.subits = 0
        self.time_so_far = 0
        print_cost = 0
        iteration_start = time.time()
        fxk = self.f.func(xk)
        rxk = self.r.func(xk)
        self.Fxk = fxk + rxk
        Fxk_best_so_far = self.Fxk
        gradfxk = self.f.grad(xk)
        self.fevals += 1
        self.gevals += 1
        inexact_pg_computation = self.config['mainsolver']['inexact_pg_computation']
        self.status = 404
        self.subsolver_consequtive_maxiter = 0
        self.subsolver_total_correct_iters = 0
        # config subsolver
        stepsize_init = None
        beta = self.config['linesearch']['beta']
        while True:
            # print(f"iteration:{self.iteration:5d} | Fxk:{self.Fxk:.3e}")
            # if self.iteration >= 51:
            #     print(self.iteration)
            # compute ipg
            xaprox, ykplus1, self.aoptim = self.r.compute_inexact_proximal_gradient_update(
                xk, self.alphak, gradfxk, self.config, yk, stepsize_init, iteration=self.iteration, xref=x_best_so_far)
            # if self.r.flag == 'lscfail':
            #     self.status = -2
            #     break
            self.time_so_far += time.time() - iteration_start - print_cost
            # print current iteration information
            self.subits += self.r.inner_its
            if self.config['mainsolver']['print_level'] > 0:
                if self.iteration % self.config['mainsolver']['print_every'] == 0:
                    self.print_header()
                self.print_iteration()
            # the abs is added avoid negetive self.r.gap due numerical cancellation
            self.optim_ub = self.aoptim + \
                np.sqrt(2 * np.abs(self.r.gap) * self.alphak)
            if save_ckpt:
                if self.config['mainsolver']['optim_scaled']:
                    milestone_scaled = milestone[0] * min(1.0, self.alphak)
                else:
                    milestone_scaled = milestone[0]
                if self.optim_ub <= milestone_scaled:
                    nnz, nz = self.r._get_group_structure(xk)
                    info = {'iteration': self.iteration, 'time': self.time_so_far,
                            'x': xk, 'F': self.Fxk, 'nnz': nnz, 'nz': nz,
                            'status': self.status,
                            'fevals': self.fevals, 'gevals': self.gevals, 'aoptim': self.aoptim,
                            'n': self.n, 'p': self.p, 'Lambda': self.r.penalty,
                            'K': self.r.K, 'subits': self.subits, 'datasetid': self.datasetid,
                            'optim': self.optim_ub
                            }
                    ckpt_tol = milestone.pop(0)
                    ckpt_dir = self.generate_ckpt_dir(save_ckpt_id, ckpt_tol)
                    np.save(ckpt_dir +
                            "/{}_info.npy".format(self.datasetname), info)
            # check termination
            if self.config['mainsolver']['optim_scaled']:
                tol_scaled = tol * min(1.0, self.alphak)
            else:
                tol_scaled = tol
            if self.optim_ub <= tol_scaled:
                self.status = 0
                break

            if self.iteration > self.config['mainsolver']['iteration_limits']:
                self.status = 1
                break
            if self.time_so_far >= self.config['mainsolver']['time_limits']:
                self.status = 2
                break
            if self.r.flag == 'correct':
                self.subsolver_total_correct_iters += 1
                if self.subsolver_total_correct_iters >= 2:
                    self.status = 4
                    break
            if self.r.flag == 'maxiter':
                self.subsolver_consequtive_maxiter += 1
                if self.subsolver_consequtive_maxiter == 2:
                    self.status = 3
                    break
            else:
                # reset counter
                self.subsolver_consequtive_maxiter = 0

            # new iteration
            iteration_start = time.time()
            self.iteration += 1
            # line search
            if self.config['mainsolver']['linesearch']:
                xtrial = xaprox
                fxtrial = self.f.func(xtrial)
                rxtrial = self.r.rxtrial
                self.bak = 0
                self.stepsize = 1
                self.d = xtrial - xk
                self.d_norm = self.aoptim
                pass_L_test = True
                # construct the upper-bound of the directional derivative
                if inexact_pg_computation == 'yd':
                    dirder_upper = -((self.d_norm**2 / self.alphak) - np.sqrt
                                     (2.0 * self.r.targap / self.alphak) * self.d_norm - self.r.targap)
                elif inexact_pg_computation == 'lee':
                    dirder_upper = - \
                        (rxk - rxtrial - np.sum(gradfxk * self.d))
                else:
                    dirder_upper = -np.inf
                const = self.config['linesearch']['eta'] * dirder_upper
                # begin backtracking
                while True:
                    if inexact_pg_computation == 'yd' or inexact_pg_computation == 'lee':
                        lhs = fxtrial - fxk + rxtrial - rxk
                        rhs = self.stepsize * const
                        # if self.r.flag == 'maxiter' or  self.iteration == 1697:
                        #     print(f"its:{self.iteration:3d} | LHS:{lhs:.4e} | RHS:{rhs:.4e} | LHS-RHS:{lhs-rhs:.4e} | dirder_upper:{dirder_upper:.4e}")
                        if lhs <= rhs:
                            self.step_take = xtrial - xk
                            xk = xtrial
                            break
                        if self.stepsize <= 1e-20:
                            # linesearch failure
                            # self.status = -1
                            # print("mainsolver: small stepsize encountered in the linesearch for alphak")
                            self.step_take = xtrial - xk
                            xk = xtrial
                            break
                        self.bak += 1
                        self.stepsize *= self.config['linesearch']['xi']
                        xtrial = xk + self.stepsize * self.d
                        fxtrial = self.f.func(xtrial)
                        rxtrial = self.r.func(xtrial)
                    else:
                        # test Lipschtiz inequality for exact solve and the schimdt inexact solve
                        l_lhs = fxtrial - fxk
                        l_rhs = np.sum(gradfxk * self.d) + 1.0 / \
                            (2.0 * self.alphak) * (self.d_norm**2)
                        if (l_lhs <= l_rhs) or (np.abs(l_lhs - l_rhs) <= 1e-15):
                            self.step_take = xtrial - xk
                            xk = xtrial
                        else:
                            # stay at the current point and reduce alphak, or equivalently enlarge L estimate
                            pass_L_test = False
                            self.alphak *= 0.8
                            fxtrial = self.f.func(xk)
                            rxtrial = rxk
                            self.step_take = 0.0
                        if self.aoptim <= 1e-8:
                            beta = 1.0
                        break
                # # terminate the whole algorithm as linesearch failed
                # if self.status == -1:
                #     break
            else:
                self.step_take = xaprox - xk
                xk = xaprox
                fxtrial = self.f.func(xk)
                rxtrial = self.r.rxtrial
                self.bak = 0
            self.step_take_size = np.sqrt(
                np.sum(self.step_take * self.step_take))
            # prepare quantities for the next dual iteration
            if self.config['subsolver']['warmstart']:
                yk = ykplus1
                stepsize_init = self.r.stepsize

            # print line search
            print_start = time.time()
            if self.config['mainsolver']['print_level'] > 0:
                self.print_linesearch()
            print_cost = time.time() - print_start

            # update parameter for two in
            if self.config['mainsolver']['linesearch'] and pass_L_test:
                if self.config['mainsolver']['stepsize_strategy'] == "frac":
                    if self.bak > 0:
                        self.alphak *= 0.8
                elif self.config['mainsolver']['stepsize_strategy'] == "model":
                    # L_k_hat makes the model decreases match with the actual decreases.
                    # print(self.iteration, self.step_take_size)
                    L_k = 1 / self.alphak
                    L_k_hat = 2 * \
                        (fxtrial - fxk - np.sum(gradfxk * self.step_take)) / \
                        (self.step_take_size**2)
                    L_k = max(2 * L_k, min(1e3 * L_k, L_k_hat))
                    L_kplus1 = max(1e-3, 1e-3 * L_k, L_k_hat)
                    self.alphak = 1.0 / L_kplus1
                elif self.config['mainsolver']['stepsize_strategy'] == "heuristic":
                    if self.bak == 0:
                        # add a safeguard
                        self.alphak = min(
                            self.alphak * beta, 100)
                    else:
                        self.alphak = max(0.8 * self.alphak, 1e-20)
                elif self.config['mainsolver']['stepsize_strategy'] == "const":
                    pass
                else:
                    raise ValueError(
                        f"Unrecognized stepsize_strategy value: {self.config['mainsolver']['stepsize_strategy']}")

            # move to new iterate
            fxk = fxtrial
            rxk = rxtrial
            self.Fxk = fxk + rxk
            gradfxk = self.f.grad(xk)
            self.fevals += 1 + self.bak
            self.gevals += 1
            if self.r.flag != 'maxiter':
                x_best_so_far = xk
                Fxk_best_so_far = self.Fxk
        if self.status != 3:
            self.solution = xk
        else:
            self.solution = x_best_so_far
            self.Fxk = Fxk_best_so_far
        self.print_exit()

        nnz, nz = self.r._get_group_structure(self.solution)
        info = {'iteration': self.iteration, 'time': self.time_so_far,
                'x': xk, 'F': self.Fxk, 'nnz': nnz, 'nz': nz,
                'status': self.status,
                'fevals': self.fevals, 'gevals': self.gevals, 'aoptim': self.aoptim,
                'n': self.n, 'p': self.p, 'Lambda': self.r.penalty,
                'K': self.r.K, 'subits': self.subits, 'datasetid': self.datasetid,
                'optim': self.optim_ub
                }
        if save_ckpt:
            ckpt_dir = self.generate_ckpt_dir(
                save_ckpt_id, self.config['mainsolver']['accuracy'])
            np.save(ckpt_dir +
                    "/{}_info.npy".format(self.datasetname), info)
        return info

    def print_problem(self):
        contents = "\n" + "=" * 80
        contents += "\n       Inexact Proximal Gradient Type Method   (version:{})  \n".format(
            self.version)
        time = datetime.datetime.now()
        contents += "=" * 80 + '\n'
        contents += f"Problem Summary: Excuted at {time.year}-{time.month}-{time.day} {time.hour}:{time.minute}\n"

        problem_attribute = self.f.__str__()
        problem_attribute += "Regularizer:{:.>56}\n".format(
            self.r.__str__())
        problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format(
            '', self.r.penalty)
        problem_attribute += "Number of groups:{:.>32}\n".format(self.r.K)
        contents += problem_attribute
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_config(self):
        contents = "\n" + "Algorithm Parameters:\n"
        contents += 'Termination Conditions:'
        contents += f" accuracy: {self.config['mainsolver']['accuracy']} | optim scaled: {self.config['mainsolver']['optim_scaled']} | time limits:{self.config['mainsolver']['time_limits']} | iteration limits:{self.config['mainsolver']['iteration_limits']}\n"
        if self.config['mainsolver']['exact_pg_computation']:
            contents += f"Evaluate proximal operator with high accuracy: {self.config['mainsolver']['exact_pg_computation_tol']}\n"
        else:
            contents += f"Inexact Strategy: {self.config['mainsolver']['inexact_pg_computation']}:"
            if self.config['mainsolver']['inexact_pg_computation'] == "schimdt":
                contents += f" delta:{self.config['inexactpg']['schimdt']['delta']:.3e} | c:{self.config['inexactpg']['schimdt']['c']:.3e}\n"
            else:
                inexact_pg_computation = self.config['mainsolver']['inexact_pg_computation']
                contents += f" gamma:{self.config['inexactpg'][inexact_pg_computation]['gamma']:.3e}\n"
        if self.config['mainsolver']['linesearch'] or self.config['subsolver']['linesearch']:
            contents += 'Lineserch Parameters:'
            contents += f" eta:{self.config['linesearch']['eta']} | xi:{self.config['linesearch']['xi']} | beta:{self.config['linesearch']['beta']}\n"
        contents += f"Proximal Stepsize update: {self.config['mainsolver']['stepsize_strategy']}\n"
        contents += "*" * 100 + "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_header(self):
        header = " Iters.   Obj.    alphak   |"
        header += self.r.print_header(self.config)
        header += self.print_linesearch_header()
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_iteration(self):
        contents = f" {self.iteration:5d} {self.Fxk:.3e} {self.alphak:.3e} |"
        contents += self.r.print_iteration(self.config)
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_linesearch_header(self):
        if not self.config['mainsolver']['linesearch']:
            contents = ""
        else:
            contents = " baks    stepsize  |dtaken| |"
        return contents

    def print_linesearch(self):
        if not self.config['mainsolver']['linesearch']:
            contents = "\n"
        else:
            contents = f"  {self.bak:3d}   {self.stepsize:.3e} {self.step_take_size:.3e} |\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_exit(self):
        contents = '\n' + "=" * 30 + '\n'
        if self.status == -2:
            contents += 'Exit: Proximal Problem Solver Failed\n'
        if self.status == -1:
            contents += 'Exit: Line Search Failed\n'
        elif self.status == 0:
            contents += 'Exit: Optimal Solution Found\n'
        elif self.status == 1:
            contents += 'Exit: Iteration limit reached\n'
        elif self.status == 2:
            contents += 'Exit: Time limit reached\n'
        elif self.status == 3:
            contents += 'Exit: Early stoppiong. (2 Consequtive subsolver maxiters).\n'
        elif self.status == 4:
            contents += 'Exit: Early stoppiong. (2 correction steps cap reached).\n'
        print(contents)
        contents += "\nFinal Results\n"
        contents += "=" * 30 + '\n'
        contents += f'Iterations:{"":.>65}{self.iteration:d}\n'
        contents += f'CPU seconds:{"":.>64}{self.time_so_far:.4f}\n'
        nnz, nz = self.r._get_group_structure(self.solution)
        contents += f'# zero groups:{"":.>62}{nz:d}\n'
        contents += f'Objective function:{"":.>57}{self.Fxk:8.6e}\n'
        contents += f'Optimality error:{"":.>59}{self.optim_ub:8.6e}\n'
        contents += f'Function evaluations:{"":.>55}{self.fevals:d}\n'
        contents += f'Gradient evaluations:{"":.>55}{self.gevals:d}\n'

        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
