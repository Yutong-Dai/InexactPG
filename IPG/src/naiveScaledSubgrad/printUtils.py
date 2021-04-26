'''
File: printUtils.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:45
Last Modified: 2021-04-18 00:04
--------------------------------------------
Description:
'''


def print_problem(problem_attribute, version, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    with open(filename, "a") as logfile:
        contents = "\n" + "=" * 80
        contents += "\n       Inexact Proximal Gradient Type Method   (version:{})  \n".format(version)
        contents += "=" * 80 + '\n'
        contents += "Problem Summary\n"
        contents += problem_attribute
        logfile.write(contents)


def print_algorithm(algodic, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    with open(filename, "a") as logfile:
        contents = "\n" + "Algorithm Parameters:\n"
        count = -1
        for k, v in algodic.items():
            if k not in ['prob', 'printlevel', 'printevery', 'max_attempts', 'version', 'delta']:
                count += 1
                contents += f" {k}: {v} "
                if count % 4 == 3:
                    contents += '\n'
                else:
                    contents += '|'
        contents += '\n' + '*' * 80 + '\n'
        logfile.write(contents)


def print_header(outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    column_titles = '  Iter      F    |   alpha       c     its    gap    epsilon  type |prox-aprox|  prox-optim/aprox-optim  p-#nz/ap-#nz  p-#z/ap-#z |  bak   stepsize  |d_full| |\n'
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


def print_iterates(iteration, F, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'

    contents = f" {iteration:5d} {F:3.3e} |"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_proximal_update(alpha, t, c, subits, gap, epsilon, eflag,
                          prox_diff, prox_optim, aprox_optim,
                          prox_nnz, aprox_nnz, prox_nz, aprox_nz, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e}  {c:2.2e} {subits:3d} {gap:2.3e} {epsilon:2.3e}  {eflag}   {prox_diff:2.3e}    {prox_optim:2.3e}/{aprox_optim:2.3e}      {prox_nnz:4d}/{aprox_nnz:4d}    {prox_nz:4d}/{aprox_nz:4d}  |"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_proximal_update_failed(alpha, t, c, subits, gap, epsilon, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e}  {c:2.2e} {subits:3d} {gap:2.3e} {epsilon:2.3e}  \n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_linesearch(d_norm, bak, stepsize, outID=None):
    # d_norm is the 2-norm of the search direction
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {bak:3d}   {stepsize:2.3e} {d_norm:2.3e} |\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_exit(status, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = '\n' + "=" * 30 + '\n'
    if status == -2:
        contents += 'Exit: Proximal Problem Solver Failed\n'
    if status == -1:
        contents += 'Exit: Line Search Failed\n'
    elif status == 0:
        contents += 'Exit: Optimal Solution Found\n'
    elif status == 1:
        contents += 'Exit: Iteration limit reached\n'
    elif status == 2:
        contents += 'Exit: Time limit reached\n'
    elif status == 3:
        contents += 'Exit: Active set identified\n'
    elif status == 4:
        contents += 'Exit: Early stop as no further progress can be made.\n'
    with open(filename, "a") as logfile:
        logfile.write(contents)
        print(contents)


def print_result(info, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = "\nFinal Results\n"
    contents += "=" * 30 + '\n'
    contents += 'Iterations:{:.>65}{:d}\n'.format("", info['iteration'])
    contents += 'CPU seconds:{:.>64}{:.4f}\n'.format("", info['time'])
    if info['nz'] is not None:
        contents += 'number of sparse groups:{:.>52}{:d}\n'.format("", info['nz'])
        contents += 'Objective function plus regularizer:{:.>40}{:8.6e}\n'.format("", info['F'])
        contents += 'Optimality error:{:.>59}{:8.6e}\n'.format("", info['optim'])
    contents += 'Function evaluations:{:.>55}{:d}\n'.format("", info['fevals'])
    contents += 'Gradient evaluations:{:.>55}{:d}\n'.format("", info['gevals'])
    contents += 'Number of backtracks:{:.>55}{:d}\n'.format("", info['baks'])
    with open(filename, "a") as logfile:
        logfile.write(contents)
