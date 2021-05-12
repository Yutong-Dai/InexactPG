'''
File: printUtils.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:45
Last Modified: 2021-04-18 17:50
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
            if k not in ['prob', 'printlevel', 'printevery', 'max_attempts', 'version', 't', 'safeguard_opt', 'safeguard_const']:
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
    column_titles = '  Iter      F    |   alpha     a_c   its    gap     epsilon    cutoff  #proj | |prox-aprox|  prox-optim/aprox-optim  p-#nz/ap-#nz  p-#z/ap-#z |\n'
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


def print_proximal_update_schimdt(alpha, alpha_update, subits, gap, epsilon, cutoff, proj,
                                  prox_diff, prox_optim, aprox_optim,
                                  prox_nnz, aprox_nnz, prox_nz, aprox_nz, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e} {alpha_update:3d}   {subits:5d} {gap:2.3e} {epsilon:2.3e}  {cutoff:2.3e} {proj:3d}  |  {prox_diff:2.3e}    {prox_optim:2.3e}/{aprox_optim:2.3e}      {prox_nnz:4d}/{aprox_nnz:4d}    {prox_nz:4d}/{aprox_nz:4d}  |\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_proximal_update_schimdt_failed(alpha, alpha_update, subits, gap, epsilon, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e} {alpha_update:3d}  -------   {subits:3d} {gap:2.3e} {epsilon:2.3e}  \n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_linesearch(bak, stepsize, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {bak:3d}   {stepsize:2.3e} |\n"
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
        contents += 'subgrad iters:{:.>55}{:d}\n'.format("", info['subgrad_iters'])
    with open(filename, "a") as logfile:
        logfile.write(contents)
