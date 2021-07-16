'''
File: printUtils.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:45
Last Modified: 2021-06-12 10:40
--------------------------------------------
Description:
'''
import datetime


def print_problem(problem_attribute, version, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    with open(filename, "a") as logfile:
        contents = "\n" + "=" * 85
        contents += "\n       Inexact Proximal Gradient Type Method   (version:{})  \n".format(version)
        time = datetime.datetime.now()
        contents += f"                        Excuted at {time.year}-{time.month}-{time.day} {time.hour}:{time.minute}\n"
        contents += "=" * 85 + '\n'
        contents += "Problem Summary\n"
        contents += problem_attribute
        logfile.write(contents)


# def print_algorithm(algodic, outID=None):
#     if outID is not None:
#         filename = '{}.txt'.format(outID)
#     else:
#         filename = 'log.txt'
#     with open(filename, "a") as logfile:
#         contents = "\n" + "Algorithm Parameters:\n"
#         count = -1
#         for k, v in algodic.items():
#             if k not in ['prob', 'printlevel', 'printevery', 'version', 'params']:
#                 count += 1
#                 contents += f" {k}: {v} "
#                 if count % 4 == 3:
#                     contents += '\n'
#                 else:
#                     contents += '|'
#         contents += '\n' + '*' * 80 + '\n'
#         logfile.write(contents)
def print_algorithm(algodic, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    with open(filename, "a") as logfile:
        contents = "\n" + "Algorithm Parameters:\n"
        contents += 'Termination Conditions:\n'
        contents += f" optimality measure: {algodic['optimality_measure']}| tol:{algodic['tol']} | maxiter:{algodic['max_iter']} | maxtime:{algodic['max_time']}\n"
        contents += 'Lineserch Parameters:\n'
        contents += f" eta:{algodic['eta']} | xi:{algodic['xi']} | zeta:{algodic['zeta']} | maxbak:{algodic['max_back']}\n"
        contents += 'Proximal Stepsize update:\n'
        contents += f" update strategy:{algodic['update_alpha_strategy']} | scale alpha for comparsion:{algodic['scale_alpha']}\n"
        contents += f"Inexact Strategy:\n"
        if algodic['inexact_type'] == 1:
            contents += f" inexact type:{algodic['inexact_type']} | gamma1:{algodic['gamma1']:1.0e}\n"
        elif algodic['inexact_type'] == 2:
            contents += f" inexact type:{algodic['inexact_type']} | gamma2:{algodic['gamma2']:1.0e} | nu:{algodic['nu']}\n"
        elif algodic['inexact_type'] == 3:
            contents += f" inexact type:{algodic['inexact_type']} | delta:{algodic['delta']} | schimdt_const:{algodic['schimdt_const']}\n"
        else:
            contents += f" inexact type:{algodic['inexact_type']} | gamma4:{algodic['gamma4']:1.0e} | nu:{algodic['nu']}\n"
        contents += f"Subsolver configuration:\n"
        contents += f" solver:{algodic['subsolver']} | warm start:{algodic['warm_start']} | verbose:{algodic['subsolver_verbose']} | maxiter:{algodic[algodic['subsolver']]['maxiter']}"
        if algodic['subsolver'] == 'projectedGD':
            contents += f"\n projectedGD init stepsize:{algodic['projectedGD']['stepsize']}\n"
        else:
            contents += '\n'
        contents += '*' * 80 + '\n'
        logfile.write(contents)


# def print_header(outID=None):
#     if outID is not None:
#         filename = '{}.txt'.format(outID)
#     else:
#         filename = 'log.txt'
#     column_titles = '  Iter      F    |   alpha     dim   subits     flag        gap       epsilon   aprox-optim   #z   #nz  |  bak   stepsize     |d|   |\n'
#     with open(filename, "a") as logfile:
#         logfile.write(column_titles)
def print_header(outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    column_titles = '  Iter      F    |   alpha     dim   subits     flag        gap       epsilon   theta     aprox-optim   #z   #nz  |  bak   stepsize     |d|   |\n'
    with open(filename, "a") as logfile:
        logfile.write(column_titles)

def print_header_lee(outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    column_titles = '  Iter      F    |   alpha     dim   subits     flag        gap       epsilon   theta     aprox-optim   #z   #nz  |  bak     |d|    |\n'
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


# def print_proximal_update(alpha, dim, subits, flag, gap, epsilon, aprox_optim, nz, nnz, outID=None):
#     """
#         pass
#     """
#     if outID is not None:
#         filename = '{}.txt'.format(outID)
#     else:
#         filename = 'log.txt'
#     contents = f" {alpha:2.3e} {dim:5d}    {subits:4d}    {flag}  {gap:+2.3e}  {epsilon:2.3e}   {aprox_optim:2.3e}  {nz:4d} {nnz:5d}  |"
#     with open(filename, "a") as logfile:
#         logfile.write(contents)

def print_proximal_update(alpha, dim, subits, flag, gap, epsilon, theta, aprox_optim, nz, nnz, outID=None):
    """
        pass
    """
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {alpha:2.3e} {dim:5d}    {subits:4d}    {flag}  {gap:+2.3e}  {epsilon:2.3e} {theta:2.3e}   {aprox_optim:2.3e}  {nz:4d} {nnz:5d}  |"
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

def print_linesearch_lee(d_norm, bak, outID=None):
    # d_norm is the 2-norm of the search direction
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = f" {bak:3d}   {d_norm:2.3e} |\n"
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
    contents += 'Sub iters :{:.>65}{:d}\n'.format("", info['subits'])
    contents += 'Sub iters Equiv :{:.>60}{:f}\n'.format("", info['subits_equiv'])
    contents += 'CPU seconds:{:.>64}{:.4f}\n'.format("", info['time'])
    if info['nz'] is not None:
        contents += 'number of sparse groups:{:.>52}{:d}\n'.format("", info['nz'])
        contents += 'Objective function:{:.>57}{:8.6e}\n'.format("", info['F'])
        contents += 'Optimality error:{:.>59}{:8.6e}\n'.format("", info['optim'])
    contents += 'Function evaluations:{:.>55}{:d}\n'.format("", info['fevals'])
    contents += 'Gradient evaluations:{:.>55}{:d}\n'.format("", info['gevals'])
    contents += 'subFunction evaluations:{:.>52}{:d}\n'.format("", info['subfevals'])
    contents += 'subGradient evaluations:{:.>52}{:d}\n'.format("", info['subgevals'])
    with open(filename, "a") as logfile:
        logfile.write(contents)


# def print_subsolver_header(probdim, subsolver, inexact_type, outter_iter, outID=None):
#     if outID is not None:
#         filename = '{}_subprob.txt'.format(outID)
#     else:
#         filename = 'log_subprob.txt'
#     column_titles = f"------- probdim: {probdim:6d} | solver:{subsolver} | inexact:{inexact_type} | outter iters:{outter_iter:6d} -------------------\n"
#     column_titles += '  Iter   |grad|    stepsize   primal    dual        gap   |  bak    stepsize     |d|   |\n'
#     with open(filename, "a") as logfile:
#         logfile.write(column_titles)

def print_subsolver_header(probdim, subsolver, inexact_type, outter_iter, outID=None):
    if outID is not None:
        filename = '{}_subprob.txt'.format(outID)
    else:
        filename = 'log_subprob.txt'
    column_titles = f"------- probdim: {probdim:6d} | solver:{subsolver} | inexact:{inexact_type} | outter iters:{outter_iter:6d} -------------------\n"
    column_titles += '  Iter   |grad|    stepsize   primal    dual        gap       theta |  bak    stepsize   |d|\n'
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


# def print_subsolver_iterates(iteration, norm_grad, beta, primal, dual, gap, bak, stepsize, norm_d, outID=None):
#     if outID is not None:
#         filename = '{}_subprob.txt'.format(outID)
#     else:
#         filename = 'log_subprob.txt'
#     if beta == '-':
#         contents = f" {iteration:5d}  {norm_grad:3.3e}  -------  {primal:3.3e} {dual:3.3e} {gap:3.3e} | {bak:3d}    {stepsize:3.3e} {norm_d:3.3e} |\n"
#     else:
#         contents = f" {iteration:5d}  {norm_grad:3.3e} {beta:3.3e} {primal:3.3e} {dual:3.3e} {gap:3.3e} | {bak:3d}    {stepsize:3.3e} {norm_d:3.3e} |\n"
#     with open(filename, "a") as logfile:
#         logfile.write(contents)
def print_subsolver_iterates(iteration, norm_grad, beta, primal, dual, gap, theta, bak, stepsize, norm_d, outID=None):
    if outID is not None:
        filename = '{}_subprob.txt'.format(outID)
    else:
        filename = 'log_subprob.txt'
    if beta == '-':
        contents = f" {iteration:5d}  {norm_grad:3.3e}  -------  {primal:3.3e} {dual:3.3e} {gap:3.3e} {theta:3.3e} | {bak:3d}    {stepsize:3.3e} {norm_d:3.3e}\n"
    else:
        contents = f" {iteration:5d}  {norm_grad:3.3e} {beta:3.3e} {primal:3.3e} {dual:3.3e} {gap:3.3e} {theta:3.3e} | {bak:3d}    {stepsize:3.3e} {norm_d:3.3e}\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)
