'''
# File: solver.py
# Project: ipg
# Created Date: 2021-08-23 11:28
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-09-09 11:33
# Modified By: Yutong Dai yutongdai95@gmail.com
#
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
# 2021-08-23	Y.D	create file. Implement algorithm.
'''
from llvmlite.binding import value
import numpy as np
import datetime
import time


class IpgSolver:
    def __init__(self, f, r, config):
        self.f = f
        self.r = r
        self.version = "0.1 (2021-08-29)"
        self.p = self.f.p
        self.config = config
        if config['mainsolver']['log']:
            self.outID = self.f.datasetName
            self.filename = '{}.txt'.format(self.outID)
        else:
            self.filename = None
        if self.config['mainsolver']['print_level'] > 0:
            self.print_problem()
            self.print_config()

    def solve(self, x_init=None, alpha_init=None):
        # configure mainsolver
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init
        if not alpha_init:
            self.alphak = 1.0
        else:
            self.alphak = alpha_init
        # dual variable
        yk = None
        self.iteration = 0
        self.fevals = 0
        self.gevals = 0
        self.time_so_far = 0
        print_cost = 0
        iteration_start = time.time()
        fxk = self.f.func(xk)
        rxk = self.r.func(xk)
        self.Fxk = fxk + rxk
        gradfxk = self.f.grad(xk)
        self.fevals += 1
        self.gevals += 1
        inexact_pg_computation = self.config['mainsolver']['inexact_pg_computation']
        # config subsolver
        stepsize_init = None
        while True:
            # check termination
            xaprox, ykplus1, self.aoptim = self.r.compute_inexact_proximal_gradient_update(
                xk, self.alphak, gradfxk, self.config, yk, stepsize_init, iteration=self.iteration)

            self.time_so_far += time.time() - iteration_start - print_cost
            # print current iteration information
            if self.config['mainsolver']['print_level'] > 0:
                if self.iteration % self.config['mainsolver']['print_every'] == 0:
                    self.print_header()
                self.print_iteration()

            if self.iteration == 0:
                tol = max(1.0, self.aoptim) * \
                    self.config['mainsolver']['accuracy']

            if self.r.aoptim <= tol:
                self.status = 0
                break

            if self.iteration > self.config['mainsolver']['iteration_limits']:
                self.status = 1
                break
            if self.time_so_far >= self.config['mainsolver']['time_limits']:
                self.status = 2
                break
            if self.r.flag == 'lscfail':
                self.status = -2
                break
            # new iteration
            iteration_start = time.time()
            self.iteration += 1
            # line search
            if self.config['mainsolver']['linesearch']:
                # construct the upper-bound of the directional derivative
                pass
            else:
                xk = xaprox
            if self.config['subsolver']['warmstart']:
                yk = ykplus1
                stepsize_init = self.r.stepsize

            # print line search
            print_start = time.time()
            if self.config['mainsolver']['print_level'] > 0:
                self.print_linesearch()
            print_cost = time.time() - print_start

            # update parameter
            if self.config['mainsolver']['stepsize_strategy'] == "frac":
                self.alphak *= self.config['linesearch']['zeta']
            elif self.config['mainsolver']['stepsize_strategy'] == "model":
                pass
            elif self.config['mainsolver']['stepsize_strategy'] == "heuristic":
                pass
            elif self.config['mainsolver']['stepsize_strategy'] == "const":
                pass
            else:
                raise ValueError(
                    f"Unrecognized stepsize_strategy value: {self.config['mainsolver']['stepsize_strategy']}")

            # move to new iterate

            fxk = self.f.func(xk)
            rxk = self.r.func(xk)
            self.Fxk = fxk + rxk
            gradfxk = self.f.grad(xk)
            self.fevals += 1
            self.gevals += 1
        self.solution = xk
        self.print_exit()

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
        contents += f" accuracy: {self.config['mainsolver']['accuracy']} | time limits:{self.config['mainsolver']['time_limits']} | iteration limits:{self.config['mainsolver']['iteration_limits']}\n"
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
            contents += f" eta:{self.config['linesearch']['eta']} | xi:{self.config['linesearch']['xi']} | zeta:{self.config['linesearch']['zeta']} | beta:{self.config['linesearch']['beta']}\n"
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

    def print_linesearch(self):
        if self.config['mainsolver']['stepsize_strategy'] == "const":
            contents = "\n"
        else:
            pass

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
            contents += 'Exit: Active set identified\n'
        elif self.status == 4:
            contents += 'Exit: Early stop as no further progress can be made.\n'
        print(contents)
        contents += "\nFinal Results\n"
        contents += "=" * 30 + '\n'
        contents += f'Iterations:{"":.>65}{self.iteration:d}\n'
        contents += f'CPU seconds:{"":.>64}{self.time_so_far:.4f}\n'
        nnz, nz = self.r._get_group_structure(self.solution)
        contents += f'# zero groups:{"":.>62}{nz:d}\n'
        contents += f'Objective function:{"":.>57}{self.Fxk:8.6e}\n'
        contents += f'Optimality error:{"":.>59}{self.aoptim:8.6e}\n'
        contents += f'Function evaluations:{"":.>55}{self.fevals:d}\n'
        contents += f'Gradient evaluations:{"":.>55}{self.gevals:d}\n'

        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
