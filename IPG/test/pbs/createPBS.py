'''
File: createPBS.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-11-16 19:42
Last Modified: 2021-01-28 00:20
--------------------------------------------
Description:
'''
import os


def create(task_name, outdir, scriptdir, node, mem, file, *argv, pbsDir='.'):
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
    contents += '\n'
    filename = '{}/{}.pbs'.format(pbsDir, task_name)
    if not os.path.exists(pbsDir):
        os.makedirs(pbsDir)
    with open(filename, "w") as pbsfile:
        pbsfile.write(contents)


if __name__ == '__main__':
    # outdir = "IPG/test/log/cache"
    # scriptdir = "IPG/test/naive"
    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for safeguard_opt in ['schimdt', 'laststep']:
    #             for safeguard_const in [1, 10]:
    #                 create(f'adpative_{loss}_{lam}_{safeguard_opt}_{safeguard_const}',
    #                        outdir,
    #                        scriptdir,
    #                        '1', '8',
    #                        'runall.py',
    #                        '--date 04_06_2021',
    #                        f'--loss {loss}',
    #                        f'--lam_shrink {lam}',
    #                        f'--safeguard_opt {safeguard_opt}',
    #                        f'--safeguard_const {safeguard_const}',
    #                        pbsDir='prelim')

    # scriptdir = "IPG/test/schimdt"
    # for lam in [0.1, 0.01]:
    #     for loss in ['logit', 'ls']:
    #         create(f'prescribe_{loss}_{lam}',
    #                outdir,
    #                scriptdir,
    #                '1', '8',
    #                'runall.py',
    #                '--date 04_06_2021',
    #                f'--loss {loss}',
    #                f'--lam_shrink {lam}',
    #                pbsDir='prelim')

    outdir = "IPG/test/log/cache"
    scriptdir = "IPG/test/preDifferentInexact"
    for loss in ['logit', 'ls']:
        for lam in [0.1, 0.01]:
            for safeguard_opt in ['schimdt', 'laststep']:
                for safeguard_const in [1, 10]:
                    create(f'naive_{loss}_{lam}_{safeguard_opt}_{safeguard_const}',
                           outdir,
                           scriptdir,
                           '1', '8',
                           'runall.py',
                           '--date 04_10_2021',
                           f'--solver naive',
                           f'--loss {loss}',
                           f'--lam_shrink {lam}',
                           f'--safeguard_opt {safeguard_opt}',
                           f'--safeguard_const {safeguard_const}',
                           pbsDir='dE')
    for loss in ['logit', 'ls']:
        for lam in [0.1, 0.01]:
            safeguard_opt = 'const'
            safeguard_const = 1e3
            create(f'naive_{loss}_{lam}_{safeguard_opt}_{safeguard_const}',
                   outdir,
                   scriptdir,
                   '1', '8',
                   'runall.py',
                   '--date 04_10_2021',
                   f'--solver naive',
                   f'--loss {loss}',
                   f'--lam_shrink {lam}',
                   f'--safeguard_opt {safeguard_opt}',
                   f'--safeguard_const {safeguard_const}',
                   pbsDir='dE')
    for lam in [0.1, 0.01]:
        for loss in ['logit', 'ls']:
            create(f'schimdt_{loss}_{lam}',
                   outdir,
                   scriptdir,
                   '1', '8',
                   'runall.py',
                   '--date 04_10_2021',
                   f'--solver schimdt',
                   f'--loss {loss}',
                   f'--lam_shrink {lam}',
                   pbsDir='dE')
    for loss in ['logit', 'ls']:
        for lam in [0.1, 0.01]:
            create(f'naive_{loss}_{lam}_exactsolve',
                   outdir,
                   scriptdir,
                   '1', '8',
                   'runall.py',
                   '--date 04_10_2021',
                   f'--solver naive',
                   f'--loss {loss}',
                   f'--lam_shrink {lam}',
                   f'--init_perturb 0',
                   f'--safeguard_opt {safeguard_opt}',
                   f'--safeguard_const {safeguard_const}',
                   pbsDir='dE')
