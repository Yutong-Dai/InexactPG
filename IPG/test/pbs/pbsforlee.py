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
    date = "10_02_2021"
    tol = 1e-5
    tol_scaled = True
    largedb = False
    max_iters = 1e5
    script_name = 'runall.py'
    mainsolver = 'lee'
    inexact_type = 'lee'
    for loss in ['logit']:
        for lam_shrink in [0.1, 0.01]:
            for group_size in [10, 100]:
                for overlap_ratio in [0.1, 0.2, 0.3]:
                    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        create(f'{inexact_type}_{loss}_{lam_shrink}_{group_size}_{overlap_ratio}_{gamma}',
                               outdir,
                               scriptdir,
                               finishdir,
                               '1', '4',
                               script_name,
                               f'--date {date}',
                               f'--loss {loss}',
                               f'--tol {tol}',
                               f'--tol_scaled {tol_scaled}',
                               f'--lam_shrink {lam_shrink}',
                               f'--group_size {group_size}',
                               f'--overlap_ratio {overlap_ratio}',
                               f'--max_time {7200}',
                               f'--mainsolver {mainsolver}',
                               f'--inexact_type {inexact_type}',
                               f'--largedb {largedb}',
                               f'--gamma_lee {gamma}')
