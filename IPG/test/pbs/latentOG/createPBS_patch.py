'''
File: createPBS.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-11-16 19:42
Last Modified: 2021-01-28 00:20
--------------------------------------------
Description:
'''
import os
import numpy as np


def create(task_name, outdir, scriptdir, finishdir, node, mem, file, *argv, pbsDir='.'):
    contents = '#PBS -N {}\n'.format(task_name)
    contents += '#PBS -e /home/yud319/InexactPG/{}/{}.err\n'.format(outdir, task_name)
    contents += '#PBS -o /home/yud319/InexactPG/{}/{}.out\n'.format(outdir, task_name)
    contents += '#PBS -l nodes={}:ppn=4\n'.format(node)
    contents += '#PBS -l mem={}gb,vmem={}gb\n'.format(mem, mem)
    contents += '#PBS -q long\n'
    contents += 'cd /home/yud319/InexactPG/{}\n'.format(scriptdir)
    contents += '/home/yud319/anaconda3/bin/python {}'.format(file)
    for arg in argv:
        contents += ' {}'.format(arg)
    # contents += '\necho `basename "$0"` >> finished.txt\n'
    contents += f'\necho "{task_name}.pbs" >> {finishdir}/finished.txt\n'
    contents += '\n'
    filename = '{}/{}.pbs'.format(pbsDir, task_name)
    if not os.path.exists(pbsDir):
        os.makedirs(pbsDir)
    with open(filename, "w") as pbsfile:
        pbsfile.write(contents)


if __name__ == '__main__':
    outdir = "IPG/test/log/cache"
    scriptdir = "IPG/test/latentOG"
    finishdir = "/home/yud319/InexactPG/IPG/test/pbs/latentOG"
    date = "06_12_2021"

    # inexact_type = 1
    # for loss in ['logit']:
    #     for lam_shrink in [0.8, 0.1, 0.05]:
    #         for group_size in [10, 100]:
    #             for overlap_ratio in [0.1, 0.3, 0.5]:
    #                 for subsolver in ['projectedGD']:
    #                     for warm_start in [True]:
    #                         for gamma1 in [1e-12]:
    #                             create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{subsolver}_{warm_start}_{gamma1}_empty',
    #                                    outdir,
    #                                    scriptdir,
    #                                    finishdir,
    #                                    '1', '4',
    #                                    'runall_patch.py',
    #                                    f'--date {date}',
    #                                    f'--loss {loss}',
    #                                    f'--lam_shrink {lam_shrink}',
    #                                    f'--group_size {group_size}',
    #                                    f'--overlap_ratio {overlap_ratio}',
    #                                    f'--inexact_type {inexact_type}',
    #                                    f'--subsolver {subsolver}',
    #                                    f'--warm_start {warm_start}',
    #                                    f'--gamma1 {gamma1}')
    inexact_type = 2
    for loss in ['logit']:
        for lam_shrink in [0.1]:
            for group_size in [10]:
                for overlap_ratio in [0.5]:
                    for subsolver in ['projectedGD']:
                        for warm_start in [True]:
                            for gamma2 in [1e-12]:
                                for nu in [0.9]:
                                    create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{subsolver}_{warm_start}_{gamma2}_{nu}',
                                           outdir,
                                           scriptdir,
                                           finishdir,
                                           '1', '4',
                                           'runall_patch.py',
                                           f'--date {date}',
                                           f'--loss {loss}',
                                           f'--lam_shrink {lam_shrink}',
                                           f'--group_size {group_size}',
                                           f'--overlap_ratio {overlap_ratio}',
                                           f'--inexact_type {inexact_type}',
                                           f'--subsolver {subsolver}',
                                           f'--warm_start {warm_start}',
                                           f'--gamma2 {gamma2}',
                                           f'--nu {nu}')
    inexact_type = 3
    for loss in ['logit']:
        for lam_shrink in [0.1]:
            for group_size in [10]:
                for overlap_ratio in [0.5]:
                    for subsolver in ['projectedGD']:
                        for warm_start in [True]:
                            for delta in [3]:
                                for schimdt_const in [1e3]:
                                    create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{subsolver}_{warm_start}_{delta}_{schimdt_const}',
                                           outdir,
                                           scriptdir,
                                           finishdir,
                                           '1', '4',
                                           'runall_patch.py',
                                           f'--date {date}',
                                           f'--loss {loss}',
                                           f'--lam_shrink {lam_shrink}',
                                           f'--group_size {group_size}',
                                           f'--overlap_ratio {overlap_ratio}',
                                           f'--inexact_type {inexact_type}',
                                           f'--subsolver {subsolver}',
                                           f'--warm_start {warm_start}',
                                           f'--delta {delta}',
                                           f'--schimdt_const {schimdt_const}')