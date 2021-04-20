'''
File: processLogFiles.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-12-20 14:41
Last Modified: 2021-03-03 10:08
--------------------------------------------
Description:
'''
from matplotlib import rc
rc('text', usetex=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plydata import *
import os
import sys


def summarize_data_to_csv(logdir, date, solver, loss, datasets, *argv, additional_cols=None, lam_shrinks=[0.1, 0.01], percents=[0.1, 0.3, 0.5], return_full=False):

    selected_cols = ['datasetid', 'status', 'time', 'iteration', 'gevals', 'F', 'nnz', 'nz']
    if additional_cols:
        selected_cols += additional_cols
    li = []
    for lam_shrink in lam_shrinks:
        for percent in percents:
            info_lst = []
            rm = []
            for datasetName in datasets:
                directory = f"{logdir}/log/{date}/{solver}/{loss}/{lam_shrink}_{percent}"
                for v in argv:
                    directory += "_{}".format(v)
                file = '{}/{}_info.npy'.format(directory, datasetName)
                try:
                    info = np.load(file, allow_pickle=True).item()
                    info_lst.append(info)
                except FileNotFoundError:
                    print(f'{file} is not found. All subsequent process will skip this dataset...')
                    rm.append(datasetName)
            for datasetName in rm:
                datasets.remove(datasetName)
            df = pd.DataFrame(info_lst)
            if return_full:
                li.append(df)
            else:
                df = df[selected_cols]
                # df['Dataset'] = datasets
                # df['shrink'] = str(float(lam_shrink))
                # df['percent'] = str(float(percent))
                # df['datasetid'] = df['Dataset'].str.cat(df['percent'], sep='_').str.cat(df['shrink'], sep="_")
                save_dir = '{}/results.csv'.format(directory)
                df.to_csv(save_dir, index=False)
    if return_full:
        frame = pd.concat(li, axis=0, ignore_index=True)
        return frame


def prepare_morales_txt(logdir, date, solver, loss, *argv, filter_condition=None, keep_header=True, solver_ext=None, overwriteFailed=True, lam_shrinks=[0.1, 0.01], percents=[0.1, 0.3, 0.5]):
    li = []
    for lam_shrink in lam_shrinks:
        for percent in percents:
            directory = f"{logdir}/log/{date}/{solver}/{loss}/{lam_shrink}_{percent}"
            for v in argv:
                directory += "_{}".format(v)
            filename = directory + '/results.csv'
            df = pd.read_csv(filename)
            li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
#     frame['datasetid'] = frame['datasetid'].str.rstrip("0")
    failed_idx = frame['status'] != 0
    if np.sum(failed_idx) != 0:
        if overwriteFailed:
            frame.loc[failed_idx, 'time'] = np.inf
            frame.loc[failed_idx, 'iteration'] = np.inf
            frame.loc[failed_idx, 'gevals'] = np.inf
        print('Algorithms terminates without finding the optimal solution in {} cases.'.format(np.sum(failed_idx)))
        if overwriteFailed:
            print('Time and Iteration are set to infity for failed cases.')
    if filter_condition is not None:
        ext = 'subset'
        frame = frame[frame['datasetid'].isin(filter_condition)]
    else:
        ext = None

    directory = "./MoralesTXT/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    original_stdout = sys.stdout
    _df = frame >> select('datasetid', 'iteration', 'time', 'gevals', 'F')
    if solver_ext:
        solver = "{}_{}".format(solver, solver_ext)
    if ext is None:
        txt_file_name = directory + '{}_{}.txt'.format(loss,solver)
    else:
        txt_file_name = directory + '{}_{}_{}.txt'.format(loss, solver, ext)
    with open(txt_file_name, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(_df.to_string(index=False, header=keep_header))
        sys.stdout = original_stdout  # Reset the standard output to its original value
    return frame, failed_idx


def sanityCheck(df1, m1, df2, m2, metric='F', threshold=0, save=False, saveDir="./"):
    cols = ['datasetid', 'F', 'nnz', 'time']
    sanity_check = pd.merge(df1[cols], df2[cols], on='datasetid', suffixes=(f'_{m1}', f'_{m2}'))
    diff = sanity_check[f'{metric}_{m1}'] - sanity_check[f'{metric}_{m2}']
    diff = diff.to_numpy()
    diff_ = diff[np.abs(diff) > threshold]
    diff_.sort()
    maskneg = diff_ < 0
    maskpos = diff_ >= 0
    plt.bar(np.arange(len(diff_))[maskneg], diff_[maskneg], color='green')
    plt.bar(np.arange(len(diff_))[maskpos], diff_[maskpos], color='red')
    plt.hlines(y=0, xmin=0 - 0.5, xmax=len(diff_) - 0.5, linewidth=1)
    plt.title(f'Metric: {metric} | green means our method is better.')
    if save:
        plt.savefig(f'{saveDir}/{m1}-{m2}.png')
    bad_id = diff > threshold
    if metric == 'F':
        good_id = diff < -threshold
        same_id = (diff >= -threshold) & (diff <= threshold)
    elif metric == 'nnz':
        good_id = diff < threshold
        same_id = diff == threshold
    else:
        raise Valuerror("Invalid metric")
    return diff_, sanity_check[bad_id], sanity_check[same_id], sanity_check[good_id]
