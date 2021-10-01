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
    contents += '#PBS -e /home/yud319/InexactPG/{}/{}.err\n'.format(
        outdir, task_name)
    contents += '#PBS -o /home/yud319/InexactPG/{}/{}.out\n'.format(
        outdir, task_name)
    contents += '#PBS -l nodes={}:ppn=2\n'.format(node)
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
    scriptdir = "IPG/test/"
    finishdir = "/home/yud319/InexactPG/IPG/test/pbs"
    # date = "09_29_2021"
    date = "09_28_2021"
    tol = 1e-5
    tol_scaled = True
    largedb = True
    use_aoptim = False
    max_time = 12 * 3600
    if use_aoptim:
        # script_name = 'runall_aoptim.py'
        script_name = 'runall_nocorrection.py'
    else:
        script_name = 'runall.py'

    # inexact_type = 'schimdt'
    # for loss in ['logit']:
    #     for lam_shrink in [0.1, 0.01]:
    #         for group_size in [10, 100]:
    #             for overlap_ratio in [0.1, 0.2, 0.3]:
    #                 clts = [1e3]
    #                 for c in clts:
    #                     create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{c}',
    #                            outdir,
    #                            scriptdir,
    #                            finishdir,
    #                            '1', '5',
    #                            script_name,
    #                            f'--date {date}',
    #                            f'--loss {loss}',
    #                            f'--tol {tol}',
    #                            f'--tol_scaled {tol_scaled}',
    #                            f'--lam_shrink {lam_shrink}',
    #                            f'--group_size {group_size}',
    #                            f'--overlap_ratio {overlap_ratio}',
    #                            f'--max_time {max_time}',
    #                            f'--inexact_type {inexact_type}',
    #                            f'--largedb {largedb}',
    #                            f'--c {c}')

    # inexact_type = 'lee'
    # for loss in ['logit']:
    #     for lam_shrink in [0.1, 0.01]:
    #         for group_size in [10, 100]:
    #             for overlap_ratio in [0.1, 0.2, 0.3]:
    #                 for gamma in [0.5]:
    #                     create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{gamma}',
    #                            outdir,
    #                            scriptdir,
    #                            finishdir,
    #                            '1', '5',
    #                            script_name,
    #                            f'--date {date}',
    #                            f'--loss {loss}',
    #                            f'--tol {tol}',
    #                            f'--tol_scaled {tol_scaled}',
    #                            f'--lam_shrink {lam_shrink}',
    #                            f'--group_size {group_size}',
    #                            f'--overlap_ratio {overlap_ratio}',
    #                            f'--max_time {max_time}',
    #                            f'--inexact_type {inexact_type}',
    #                            f'--largedb {largedb}',
    #                            f'--gamma_lee {gamma}')
    inexact_type = 'yd'
    for loss in ['logit']:
        for lam_shrink in [0.1, 0.01]:
            for group_size in [10, 100]:
                for overlap_ratio in [0.1, 0.2, 0.3]:
                    for gamma in [0.2]:
                        create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{gamma}',
                               outdir,
                               scriptdir,
                               finishdir,
                               '1', '5',
                               script_name,
                               f'--date {date}',
                               f'--loss {loss}',
                               f'--tol {tol}',
                               f'--tol_scaled {tol_scaled}',
                               f'--lam_shrink {lam_shrink}',
                               f'--group_size {group_size}',
                               f'--overlap_ratio {overlap_ratio}',
                               f'--max_time {max_time}',
                               f'--inexact_type {inexact_type}',
                               f'--largedb {largedb}',
                               f'--gamma_yd {gamma}')
