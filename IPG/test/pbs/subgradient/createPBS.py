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
    outdir = "IPG/test/log/cache"
    scriptdir = "IPG/test/subgradient"
    # date = "04_18_2021"
    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for safeguard_opt in ['schimdt']:
    #             for safeguard_const in [1e8, 1e11]:
    #                 for t in [1e-12, 1e-3, 1]:
    #                     create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                            outdir,
    #                            scriptdir,
    #                            '1', '4',
    #                            'runall.py',
    #                            f'--date {date}',
    #                            f'--solver naive',
    #                            f'--loss {loss}',
    #                            f'--lam_shrink {lam}',
    #                            f'--t {t}',
    #                            f'--safeguard_opt {safeguard_opt}',
    #                            f'--safeguard_const {safeguard_const}')
    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for safeguard_opt in ['schimdt', 'laststep']:
    #             for safeguard_const in [1e-1, 1, 10]:
    #                 for t in [1e-12, 1e-3, 1]:
    #                     create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                            outdir,
    #                            scriptdir,
    #                            '1', '4',
    #                            'runall.py',
    #                            f'--date {date}',
    #                            f'--solver naive',
    #                            f'--loss {loss}',
    #                            f'--lam_shrink {lam}',
    #                            f'--t {t}',
    #                            f'--safeguard_opt {safeguard_opt}',
    #                            f'--safeguard_const {safeguard_const}')
    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         safeguard_opt = 'const'
    #         safeguard_const = 1e3
    #         for t in [1e-12, 1e-3, 1]:
    #             create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                    outdir,
    #                    scriptdir,
    #                    '1', '4',
    #                    'runall.py',
    #                    f'--date {date}',
    #                    f'--solver naive',
    #                    f'--loss {loss}',
    #                    f'--lam_shrink {lam}',
    #                    f'--t {t}',
    #                    f'--safeguard_opt {safeguard_opt}',
    #                    f'--safeguard_const {safeguard_const}')
    # for lam in [0.1, 0.01]:
    #     for loss in ['logit', 'ls']:
    #         for schimdt_const in [1e-1, 1, 1e1]:
    #             create(f'schimdt_{loss}_{lam}_{schimdt_const}',
    #                    outdir,
    #                    scriptdir,
    #                    '1', '4',
    #                    'runall.py',
    #                    f'--date {date}',
    #                    f'--solver schimdt',
    #                    f'--loss {loss}',
    #                    f'--lam_shrink {lam}',
    #                    f'--schimdt_const {schimdt_const}')
    # date = "04_23_2021"

    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         safeguard_opt = 'const'
    #         safeguard_const = 1e3
    #         for t in [1e-12]:
    #             create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                    outdir,
    #                    scriptdir,
    #                    '1', '4',
    #                    'runall.py',
    #                    f'--date {date}',
    #                    f'--solver naive',
    #                    f'--loss {loss}',
    #                    f'--lam_shrink {lam}',
    #                    f'--t {t}',
    #                    f'--safeguard_opt {safeguard_opt}',
    #                    f'--safeguard_const {safeguard_const}')
    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for schimdt_const in [1, 1e3, 1e5, 1e11]:
    #             create(f'schimdt_{loss}_{lam}_{schimdt_const}',
    #                    outdir,
    #                    scriptdir,
    #                    '1', '4',
    #                    'runall.py',
    #                    f'--date {date}',
    #                    f'--solver schimdt',
    #                    f'--loss {loss}',
    #                    f'--lam_shrink {lam}',
    #                    f'--schimdt_const {schimdt_const}')

    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for safeguard_opt in ['schimdt']:
    #             for safeguard_const in [1, 1e3, 1e5, 1e11]:
    #                 for t in [1e-12]:
    #                     create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                            outdir,
    #                            scriptdir,
    #                            '1', '4',
    #                            'runall.py',
    #                            f'--date {date}',
    #                            f'--solver naive',
    #                            f'--loss {loss}',
    #                            f'--lam_shrink {lam}',
    #                            f'--t {t}',
    #                            f'--safeguard_opt {safeguard_opt}',
    #                            f'--safeguard_const {safeguard_const}')

    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for safeguard_opt in ['laststep']:
    #             for safeguard_const in [1e-1, 1, 10, 1e3]:
    #                 for t in [1e-12]:
    #                     create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                            outdir,
    #                            scriptdir,
    #                            '1', '4',
    #                            'runall.py',
    #                            f'--date {date}',
    #                            f'--solver naive',
    #                            f'--loss {loss}',
    #                            f'--lam_shrink {lam}',
    #                            f'--t {t}',
    #                            f'--safeguard_opt {safeguard_opt}',
    #                            f'--safeguard_const {safeguard_const}')
    # for loss in ['logit', 'ls']:
    #     for lam in [0.1, 0.01]:
    #         for safeguard_opt in ['none']:
    #             safeguard_const = np.inf
    #             for t in [1e-12]:
    #                 create(f'naive_{loss}_{lam}_{t}_{safeguard_opt}_{safeguard_const}',
    #                        outdir,
    #                        scriptdir,
    #                        '1', '4',
    #                        'runall.py',
    #                        f'--date {date}',
    #                        f'--solver naive',
    #                        f'--loss {loss}',
    #                        f'--lam_shrink {lam}',
    #                        f'--t {t}',
    #                        f'--safeguard_opt {safeguard_opt}',
    #                        f'--safeguard_const {safeguard_const}')
    date = "04_23_2021"
    for loss in ['logit', 'ls']:
        for lam in [0.1, 0.01]:
            for schimdt_const in [1e1]:
                create(f'schimdt_{loss}_{lam}_{schimdt_const}',
                       outdir,
                       scriptdir,
                       '1', '4',
                       'runall.py',
                       f'--date {date}',
                       f'--solver schimdt',
                       f'--loss {loss}',
                       f'--lam_shrink {lam}',
                       f'--schimdt_const {schimdt_const}')
