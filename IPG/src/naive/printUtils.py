'''
File: printUtils.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 16:45
Last Modified: 2021-03-22 16:51
--------------------------------------------
Description:
'''


def print_problem(method, provide_initial, version, Lambda, K, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    with open(filename, "a") as logfile:
        contents = "\n" + "=" * 80
        contents += "\n          Proximal Gradient Type Method   (version:{})  \n".format(version)
        contents += "=" * 80
        contents += "\nMethod:{:.>48}{}\n".format("", method)
        contents += "Provide Initial:{:.>39}{}\n".format("", provide_initial)
        contents += "Penalty Parameter:{:.>37}{:3.4e}\n".format('', Lambda)
        contents += "Number of groups:{:.>40}\n".format(K)
        contents += "\n"
        logfile.write(contents)


def print_header_PG(outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    column_titles = '  Iter            F            optim        bak     alpha\n'
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


def print_iterates_PG(iteration, F, optim, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    if isinstance(optim, str):
        contents = " {:5d}       {:3.6e}   ----------- |".format(iteration, F)
    else:
        contents = " {:5d}       {:3.6e}   {:3.5e} |".format(iteration, F, optim)
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_update_PG(bak, alpha, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = " {:3d}    {:3.4e}   |\n".format(bak, alpha)
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_cg_step(typeOfIteration, nS, gradF_norm, subprobFlag, subits, res, res_target, normd, cg_type, newzb, dirder, proj, j, stepsize,
                  prox_time=None, newton_time=None, ls_time=None, cg_time=None, outID=None, print_time=False):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = " {ctype:>3s} {nS:>5d} {gradF_norm:8.2e} {subpb:5s} {subits:>4d} {res:8.2e} {res_target:8.2e} {normd:8.2e} | {itype:4s}  {newzb:>4d}  {dirder: 8.2e} {proj:>2d}/{j:>2d}  {s:8.2e}| ".format(
        ctype=typeOfIteration, nS=nS, gradF_norm=gradF_norm, subpb=subprobFlag, subits=subits, res=res, res_target=res_target, normd=normd,
        itype=cg_type, newzb=newzb, dirder=dirder, proj=proj, j=j, s=stepsize)
    if print_time:
        contents += " {:3.2e}    {:3.2e}   {:3.2e}   --------   {:3.2e}|\n".format(prox_time, newton_time, ls_time, cg_time)
    else:
        contents += "\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_pg_step(typeOfIteration, nS, subits, normd, j, stepsize,
                  prox_time=None, ls_time=None, pg_time=None, outID=None, print_time=False):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = " {ctype:>3s} {nS:>5d} -------- ----- {subits:>4d} -------- -------- {normd:8.2e} | desc  ----- ---------   /{j:>2d}  {s:8.2e}| ".format(
        ctype=typeOfIteration, nS=nS, subits=subits, normd=normd, j=j, s=stepsize)
    if print_time:
        contents += " {:3.2e}    --------   {:3.2e}   {:3.2e}   --------|\n".format(prox_time, ls_time, pg_time)
    else:
        contents += "\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_exit(status, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = '\n' + "=" * 30 + '\n'
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
    contents += 'PG iters:{:.>67}{:d}\n'.format("", info['num_pg_steps'])
    contents += 'CG desc iters:{:.>62}{:d}\n'.format("", info['num_cgdesc_steps'])
    contents += 'CG proj iters:{:.>62}{:d}\n'.format("", info['num_cg0_stpes'])
    contents += 'CPU seconds:{:.>64}{:.4f}\n'.format("", info['time'])
    contents += 'Proximal gradient time:{:.>53}{:3.2e}\n'.format("", info['prox_time'])
    contents += 'CG iteration time:{:.>58}{:3.2e}\n'.format("", info['cg_time'])
    contents += 'PG iteration time:{:.>58}{:3.2e}\n'.format("", info['pg_time'])
    contents += 'Newton-CG time:{:.>61}{:3.2e}\n'.format("", info['newton_time'])
    contents += 'CG linesearch time:{:.>57}{:3.2e}\n'.format("", info['cg_ls_time'])
    contents += 'PG linesearch time:{:.>57}{:3.2e}\n'.format("", info['pg_ls_time'])
    contents += 'Step seconds:{:.>63}{:.4f}\n'.format("", info['time_update_stepsize'])
    contents += 'number of sparse groups:{:.>52}{:d}\n'.format("", info['nz'])
    contents += 'Objective function:{:.>57}{:8.6e}\n'.format("", info['f'])
    contents += 'Objective function plus regularizer:{:.>40}{:8.6e}\n'.format("", info['F'])
    contents += 'Optimality error:{:.>59}{:8.6e}\n'.format("", max(info['chipg'], info['chicg']))
    contents += 'Function evaluations:{:.>55}{:d}\n'.format("", info['fevals'])
    contents += 'Gradient evaluations:{:.>55}{:d}\n'.format("", info['gevals'])
    contents += 'Hessian vector products:{:.>52}{:d}\n\n\n'.format("", info['HvProds'])
    with open(filename, "a") as logfile:
        logfile.write(contents)
