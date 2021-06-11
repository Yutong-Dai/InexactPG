'''
File: printUtils.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:45
Last Modified: 2021-06-07 23:09
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
            if k not in ['prob', 'printlevel', 'printevery', 'max_attempts', 'version', 'params']:
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
    column_titles = '  Iter      f    |   alpha     dim   subits   flag        gap       epsilon   theta    proj  aprox-optim   #z   #nz  |  bak   stepsize  |d_full| |\n'
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


def print_proximal_update(alpha, dim, subits, flag, gap, epsilon, theta, projection, aprox_optim, nz, nnz, outID=None):
    """
        projection: number of correction two feasibility performed
    """
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e} {dim:5d}    {subits:3d}    {flag}  {gap:+2.3e}  {epsilon:2.3e} {theta:2.3e}  {projection:3d}   {aprox_optim:2.3e}  {nz:4d} {nnz:5d}  |"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_proximal_update_failed(alpha, dim, subits, flag, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e}  {dim:5d}    {subits:3d}   {flag}  \n"
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
        contents += 'Objective function(f):{:.>54}{:8.6e}\n'.format("", info['f'])
        contents += 'Optimality error:{:.>59}{:8.6e}\n'.format("", info['optim'])
    contents += 'Function evaluations:{:.>55}{:d}\n'.format("", info['fevals'])
    contents += 'Gradient evaluations:{:.>55}{:d}\n'.format("", info['gevals'])
    with open(filename, "a") as logfile:
        logfile.write(contents)
