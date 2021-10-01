import sys
import glob
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
def create_paths(logdir='../IPG/test/log', date='09_10_2021', inexact_type='lee', loss='logit',
                 use_ckpt=True, tol = 1e-6, lam_shrink=[0.1], group_size=[10], overlap_ratio=[0.1], excludes=None, **kwargs):
    list_all_npy_path = []
    for lam in lam_shrink:
        for grp in group_size:
            for r in overlap_ratio:
                if use_ckpt:
                    minimal_dir = f'{logdir}/{date}/{inexact_type}/{loss}/ckpt/{tol}/{lam}_{grp}_{r}'
                else:
                    minimal_dir = f'{logdir}/{date}/{inexact_type}/{loss}/logfile/{lam}_{grp}_{r}'
                for key, value in kwargs.items():
                    minimal_dir += f'_{kwargs[key]}'
                datasets = glob.glob(f'{minimal_dir}/*.npy')
                datasets.sort()
                datasets_rmed = datasets.copy()
                if excludes is not None:
                    for e in excludes:
                        for p in datasets:
                            datasetname = p.split("/")[-1].split("_")[0]
                            if datasetname == e:
                                datasets_rmed.remove(p)
                list_all_npy_path += datasets_rmed
    return list_all_npy_path
def load_df_from_paths(list_all_npy_path, cols=['datasetid', 'status', 'time', 'iteration', 'subits',
                                                'F', 'optim', 'nz', 'nnz']):
    info_lst = []
    for path in list_all_npy_path:
        info = np.load(path, allow_pickle=True).item()
        info_lst.append(info)
    try:
        df = pd.DataFrame(info_lst)[cols]
    except KeyError:
        cols_ = cols.copy()
        cols_[cols_.index('optim')] = 'aoptim'
        df = pd.DataFrame(info_lst)[cols_]
    # summarize status
    codes = df['status'].unique()
    formatter = f'{len(str(df.shape[0]))}d'
    for code in codes:
        count = np.sum(df['status'] == code)
        if count is None:
            count = 0
        print(f" {count:{formatter}}/{df.shape[0]} instances terminate with status: {code:2d}")
    return df

# def get_all(logdir, date, inexact_type, loss, use_ckpt, tol,
#             lam_shrink, group_size, overlap_ratio,
#             excludes=None, param_lst=None):
#     algo_df_dict = {}
#     for p in param_lst:
#             algorithm = f'{inexact_type}-{p}'
#             print(f'{algorithm}')
#             paths = create_paths(logdir, date, inexact_type, loss, use_ckpt, tol, 
#                                  lam_shrink, group_size, overlap_ratio, excludes, p=p)
#             if paths == []:
#                 print(' empty')
#                 df = None
#             else:
#                 df = load_df_from_paths(paths)
#             algo_df_dict[algorithm] = df
#     return algo_df_dict

def get_all(logdir, date, inexact_type, loss, use_ckpt, tol,
            lam_shrink, group_size, overlap_ratio,
            excludes=None, param_lst=None):
    algo_df_dict = {}
    for p in param_lst:
        algorithm = f'{inexact_type}-{p}'
        print(f'{algorithm}')
        paths = []
        for date_ in date:
            paths_ = create_paths(logdir, date_, inexact_type, loss, use_ckpt, tol, 
                                 lam_shrink, group_size, overlap_ratio, excludes, p=p)
            paths += paths_
        if paths == []:
            print(' empty')
            df = None
        else:
            df = load_df_from_paths(paths)
        algo_df_dict[algorithm] = df
    return algo_df_dict

class PerformanceProfile:
    def __init__(self, algo_df_dic, failcode=-2, overwriteFailed=True):
        self.algo_lst = [*algo_df_dic.keys()]
        self.algo_df_dic = deepcopy(algo_df_dic)
        all_failed = algo_df_dic[self.algo_lst[0]]['datasetid'].to_numpy()
        for algo in self.algo_lst:
            frame = self.algo_df_dic[algo]
            if overwriteFailed:
                failed_idx = frame['status'] == failcode
                all_failed = np.intersect1d(all_failed, frame[failed_idx]['datasetid'].to_numpy())
                frame.loc[failed_idx, 'time'] = np.inf
                frame.loc[failed_idx, 'iteration'] = np.inf
                frame.loc[failed_idx, 'subgrad_iters'] = np.inf
                frame.loc[failed_idx, 'nnz'] = np.inf
        print(f"All algorithms failed in {len(all_failed)} instances (failure code {failcode})")
        self.all_failed = all_failed
        if overwriteFailed:
            print("Metrics for failed instances are overwritten with np.inf")

    def get_subset_by_time(self, threshold=1, remove_failed=True):
        """
        inplace modification
        """
        filter_condition = np.array([])
        for algo in self.algo_lst:
            frame = self.algo_df_dic[algo]
            qualified_datasetid = frame[frame['time'] >= threshold]['datasetid'].to_numpy()
            filter_condition = np.union1d(filter_condition, qualified_datasetid)
        if remove_failed:
            filter_condition = np.setdiff1d(filter_condition, self.all_failed)
        filter_condition = np.sort(filter_condition)
        for algo in self.algo_lst:
            frame = self.algo_df_dic[algo]
            self.algo_df_dic[algo] = frame[frame['datasetid'].isin(filter_condition)]
        # perform sanity check
        base = self.algo_df_dic[self.algo_lst[0]]
        num_records = base.shape[0]
        datasetid = base['datasetid'].to_numpy()
        for i in range(1, len(self.algo_lst)):
            frame = self.algo_df_dic[self.algo_lst[i]]
            datasetid_candidate = frame['datasetid'].to_numpy()
            if frame.shape[0] != num_records:
                msg = f'Algortithm {self.algo_lst[0]} has {num_records} records.\n'
                msg += f'Algortithm {self.algo_lst[i]} has {frame.shape[0]} records.\n'
                msg += 'Inconsistent!'
                raise ValueError(msg)
            for j in range(len(datasetid)):
                if datasetid[j] != datasetid_candidate[j]:
                    msg = f"compare {datasetid[i]} with {frame['datasetid'][i]} \n"
                    msg += 'unmatched instance'
                    raise ValueError(msg)
        print(f"After subsetting, {num_records} instances are kept.")

    def plot(self, column, options={}, save=False, saveDir='./', ext=None, dpi=300, 
    show_num=False, use_tt=True, plot=True, auc=True, factor=1.5, format='pdf', labels=None):
        if 'color' not in options.keys():
            options['color'] = 'rgb'
        if 'ratio_max' not in options.keys():
            options['ratio_max'] = 1
        self.options = options
        self.num_algo = len(self.algo_lst)
        pools = {}
        for i in range(self.num_algo - 1):
            for j in range(i + 1, self.num_algo):
                algo1 = self.algo_lst[i]
                algo2 = self.algo_lst[j]
                if labels is not None:
                    algo1_label = labels[i]
                    algo2_label = labels[j]
                data1 = self.algo_df_dic[algo1][column].copy()
                data2 = self.algo_df_dic[algo2][column].copy()
                data1.loc[data1 < 0] = np.inf
                data2.loc[data2 < 0] = np.inf
                ratio = (data1 / data2).to_numpy()
                ratio[ratio == 0] = 1e-16
                ratio = -np.log2(ratio)
                ratio[ratio == -np.log2(1e-16)] = np.inf
                ratio = ratio[~np.isnan(ratio)]
                bars_pos = np.zeros(ratio.shape)
                bars_neg = np.zeros(ratio.shape)
                flag = ratio > 0
                bars_pos[flag] = ratio[flag]
                bars_neg[~flag] = ratio[~flag]
                if auc:
                    posmax = np.max(bars_pos[bars_pos != np.inf])
                    negmin = np.min(bars_neg[bars_neg != -np.inf])
                    ratio_max = max(posmax, -negmin)
                    bars_pos[bars_pos == np.inf] = ratio_max * factor
                    bars_neg[bars_neg == -np.inf] = -ratio_max * factor
                    self.bars_pos = bars_pos
                    self.bars_neg = bars_neg
                else:
                    bars_pos[bars_pos == np.inf] = self.options['ratio_max']
                    bars_neg[bars_neg == -np.inf] = -self.options['ratio_max']
                # sort in descending order
                bars_pos[::-1].sort()
                bars_neg[::-1].sort()
                if auc:
                    win_auc = np.around(np.sum(bars_pos), 3)
                    lose_auc = np.around(np.abs(np.sum(bars_neg)), 3)
                win = sum(flag)
                lose = sum(ratio < 0)
                if plot:
                    figure = plt.figure()
                    if show_num:
                        if labels is None:
                            label_1 = algo1 + ' {}'.format(win)
                            label_2 = algo2 + ' {}'.format(lose)
                        else:
                            label_1 = algo1_label + ' {}'.format(win)
                            label_2 = algo2_label + ' {}'.format(lose)
                        if auc:
                            label_1 += f' auc:{win_auc}'
                            label_2 += f' auc:{lose_auc}'
                    else:
                        if labels is None:
                            label_1 = algo1
                            label_2 = algo2
                        else:
                            label_1 = algo1_label
                            label_2 = algo2_label
                        if auc:
                            label_1 += f' auc:{win_auc}'
                            label_2 += f' auc:{lose_auc}'
                        print('{} Win:{} | Lose:{}'.format(algo1, win, lose))
                    if use_tt:
                        label_1 = r'\texttt{{{}}}'.format(label_1)
                        label_2 = r'\texttt{{{}}}'.format(label_2)
                    if self.options['color'] == 'bw':
                        plt.bar(range(len(bars_pos)), bars_pos, color=[32 / 225, 32 / 225, 32 / 225, 1], label=label_1, width=1)
                        plt.bar(range(len(bars_neg)), bars_neg, color=[192 / 225, 192 / 225, 192 / 225, 1], label=label_2, width=1)
                    else:
                        plt.bar(range(len(bars_pos)), bars_pos, color='b', label=label_1, width=1)
                        plt.bar(range(len(bars_neg)), bars_neg, color='r', label=label_2, width=1)
                    plt.xlim(-1, len(bars_pos) + 1)
                    plt.ylim(-self.options['ratio_max'], self.options['ratio_max'])
                    plt.legend()
                    plt.hlines(y=0, xmin=0 - 0.5, xmax=len(bars_pos) - 0.5, linewidth=1)
                    plt.ylabel('-log2(ratio)', fontsize=13)

                    if column == 'time':
                        text = 'computational time'
                    elif column == 'fevals':
                        text = 'function evaluations'
                    elif column == 'iteration':
                        text = 'iterations'
                    elif column == 'subgrad_iters':
                        text = 'subgradient iterations'
                    else:
                        text = column
                    if auc:
                        plt.title("Metric: {} (Area Under the Curve)".format(text))
                    else:
                        plt.title("Metric: {}".format(text))
                    if save:
                        if ext is None:
                            filename = saveDir + '/{}-{}_{}.{}'.format(algo1, algo2, column, format)
                        else:
                            filename = saveDir + '/{}-{}_{}_{}.{}'.format(algo1, algo2, column, ext, format)
                        # metrics
                        if dpi:
                            figure.savefig(filename, dpi=dpi)
                        else:
                            figure.savefig(filename)
                else:
                    key = f'{algo1}_{algo2}'
                    if auc:
                        pools[key] = (win_auc, lose_auc, algo1, algo2)
                    else:
                        pools[key] = (win, lose, algo1, algo2)
        return pools


def get_best(pools):
    copy_pools = deepcopy(pools)
    iters = 0
    max_iters = int(np.log2(len(pools))) + 3
    while len(copy_pools) > 1 and iters <= max_iters:
        iters += 1
        cand = []
        for v in copy_pools.values():
            if v[0] > v[1]:
                cand.append(v[2])
            else:
                cand.append(v[3])
        cand = list(set(cand))
        copy_pools = {}
        for i in range(len(cand) - 1):
            for j in range(i + 1, len(cand)):
                key = f'{cand[i]}_{cand[j]}'
                if key in pools.keys():
                    copy_pools[key] = pools[key]
                else:
                    key = f'{cand[j]}_{cand[i]}'
                    copy_pools[key] = pools[key]
    if iters < max_iters:
        val = [*copy_pools.values()][0]
        if val[0] >= val[1]:
            return val[2]
        else:
            return val[3]
    else:
        print("no stric ordering, choose max instances")
        max_so_far = -1
        for val in pools.values():
            curmax = max(val[0], val[1])
            if curmax > max_so_far:
                max_so_far = curmax
                if val[0] >= val[1]:
                    max_key = val[2]
                else:
                    max_key = val[3]
        return max_key
    
def collect_time(df_dict):
    lst = []
    for key in df_dict.keys():
        df_ = df_dict[key]
        time_ = np.sum(df_['time'])
        lst.append(time_)
    return lst
def prepare_box(df_dict):
    lst = []
    for key in df_dict.keys():
        df_ = df_dict[key]
        time_ = df_['time'].to_numpy()
        lst.append(time_)
    return lst

def pair_wise_comparison(df1, df2, suffixes_lst):
    df12_merged = pd.merge(df1, df2, on='datasetid', 
                           suffixes=(suffixes_lst[0], suffixes_lst[1]), how='left')
    better12_z = np.sum((df12_merged[f'nz{suffixes_lst[0]}'] - df12_merged[f'nz{suffixes_lst[1]}']) > 0 )
    same12_z = np.sum((df12_merged[f'nz{suffixes_lst[0]}'] - df12_merged[f'nz{suffixes_lst[1]}']) == 0 ) 
    worse12_z = np.sum((df12_merged[f'nz{suffixes_lst[0]}'] - df12_merged[f'nz{suffixes_lst[1]}']) < 0 ) 
    
    better12_F = np.sum((df12_merged[f'F{suffixes_lst[0]}'] - df12_merged[f'F{suffixes_lst[1]}']) < -1e-8 )
    same12_F = np.sum(np.abs(df12_merged[f'F{suffixes_lst[0]}'] - df12_merged[f'F{suffixes_lst[1]}']) <=1e-8 ) 
    worse12_F = np.sum((df12_merged[f'F{suffixes_lst[0]}'] - df12_merged[f'F{suffixes_lst[1]}']) > 1e-8 )     

    print(f"For {suffixes_lst[0]}-{suffixes_lst[1]} comparsion:\n==========================")
    print(" In terms final F:")
    print(f"  better:{better12_F} | same:{same12_F} | worse: {worse12_F}")
    print(" In terms #z:")
    print(f"  better:{better12_z} | same:{same12_z} | worse: {worse12_z}")
    return better12_z, same12_z, worse12_z, better12_F, same12_F, worse12_F