import sys
import glob
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


def create_paths(logdir='../IPG/test/log', date='04_18_2021', solver='naive', loss='logit', lam_shrink=[0.1, 0.01], percent=[0.1, 0.2], *argv):
    list_all_npy_path =[]
    for lam in lam_shrink:
        for per in percent:
            minimal_dir = f'{logdir}/{date}/{solver}/{loss}/{lam}_{per}'
            for arg in argv:
                minimal_dir += f'_{arg}'
            datasets = glob.glob(f'{minimal_dir}/*.npy')
            datasets.sort()
            list_all_npy_path += datasets
    return list_all_npy_path

def load_df_from_paths(list_all_npy_path, cols=['datasetid', 'status', 'time', 'iteration', 'F', 'optim','subgrad_iters']):
    info_lst = []
    for path in list_all_npy_path:
        info = np.load(path, allow_pickle=True).item()
        info_lst.append(info)
    df = pd.DataFrame(info_lst)[cols]
    # summarize status
    codes = df['status'].unique()
    formatter = f'{len(str(df.shape[0]))}d'
    for code in codes:
        count = np.sum(df['status'] == code)
        print(f" {count:{formatter}}/{df.shape[0]} instances terminate with status: {code:2d}")
    return df

def get_all_naive(logdir, date, loss, lam_shrink, percent, safeguard):
    algo_df_dict = {}
    solver = 'naive'
    if safeguard == 'const':
        safeguard_const = 1e3
        for t in [1e-12, 1e-3, 1e0]:
            algorithm = f'{solver}-{t}-{safeguard}-{safeguard_const}'
            print(f'{algorithm}')
            paths = create_paths(logdir, date, solver, loss, lam_shrink, percent, t, 'const', 1e3)
            if paths == []:
                print(' empty')
                df = None
            else:
                df = load_df_from_paths(paths)
            algo_df_dict[algorithm] = df
    else:
        for safeguard_const in [1e-1, 1e0, 1e1]:
            for t in [1e-12, 1e-3, 1e0]:
                algorithm = f'{solver}-{t}-{safeguard}-{safeguard_const}'
                print(f'{algorithm}')
                paths = create_paths(logdir, date, solver, loss, lam_shrink, percent, t, safeguard, safeguard_const)
                if paths == []:
                    print(' empty')
                    df = None
                else:
                    df = load_df_from_paths(paths)
                algo_df_dict[algorithm] = df
    return algo_df_dict

def get_all_schimdt(logdir, date, loss, lam_shrink, percent):
    solver = 'schimdt'
    algo_df_dict = {}
    for schimdt_const in [1e-1, 1e0, 1e1]:
        algorithm = f"{solver}-{'none'}-{schimdt_const}"
        print(f'{algorithm}')
        paths = create_paths(logdir, date, 'schimdt', loss, lam_shrink, percent, "'none'", schimdt_const)
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
        all_failled = algo_df_dic[self.algo_lst[0]]['datasetid'].to_numpy()
        for algo in self.algo_lst:
            frame = self.algo_df_dic[algo] 
            if overwriteFailed:
                failed_idx = frame['status'] == failcode
                all_failled = np.intersect1d(all_failled, frame[failed_idx]['datasetid'].to_numpy())
                frame.loc[failed_idx, 'time'] = np.inf
                frame.loc[failed_idx, 'iteration'] = np.inf
                frame.loc[failed_idx, 'subgrad_iters'] = np.inf
        print(f"All algorithms failed in {len(all_failled)} instances")
        self.all_failled = all_failled
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
            filter_condition = np.setdiff1d(filter_condition, self.all_failled)
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
        print(f"After subsetting, {num_records} instabces are kept.")
    def plot(self, column, options={}, save=False, saveDir='./', ext=None, dpi=300, show_num=False, use_tt=True, plot=True):
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
                data1 = self.algo_df_dic[algo1][column].copy()
                data2 = self.algo_df_dic[algo2][column].copy()
                data1.loc[data1 < 0] = np.inf
                data2.loc[data2 < 0] = np.inf
                ratio = (data1 / data2).to_numpy()
                ratio[ratio == 0] = 1e-16
                ratio = -np.log2(ratio)
                ratio = ratio[~np.isnan(ratio)]
                bars_pos = np.zeros(ratio.shape)
                bars_neg = np.zeros(ratio.shape)
                flag = ratio > 0
                bars_pos[flag] = ratio[flag]
                bars_neg[~flag] = ratio[~flag]
                bars_pos[bars_pos == np.inf] = self.options['ratio_max']
                bars_neg[bars_neg == -np.inf] = -self.options['ratio_max']
                # sort in descending order
                bars_pos[::-1].sort()
                bars_neg[::-1].sort()
                win = sum(flag)
                lose = sum(ratio < 0)
                if plot:
                    figure = plt.figure()
                    if show_num:
                        label_1 = algo1 + ' {:3d}'.format(win)
                        label_2 = algo2 + ' {:3d}'.format(lose)
                    else:
                        label_1 = self.algo_lst[algo1]
                        label_2 = self.algo_lst[algo2]
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
                    plt.title("Metric: {}".format(text))
                    if save:
                        if ext is None:
                            filename = saveDir + '{}-{}_{}.eps'.format(algo1, algo2, column)
                        else:
                            filename = saveDir + '{}-{}_{}_{}.eps'.format(algo1, algo2, column, ext)
                        # metrics
                        if dpi:
                            figure.savefig(filename, dpi=dpi)
                        else:
                            figure.savefig(filename)
                else:
                    key = f'{algo1}_{algo2}'
                    pools[key] = (win, lose, algo1, algo2)
        return pools

def get_best(pools):
    copy_pools = deepcopy(pools)
    iters = 0
    max_iters = int(np.log2(len(pools))) + 3
    while len(copy_pools)>1 and iters <= max_iters:
        iters += 1
        cand = []
        for v in copy_pools.values():
            if v[0]>v[1]:
                cand.append(v[2])
            else:
                cand.append(v[3])
        cand = list(set(cand))
        copy_pools = {}
        for i in range(len(cand)-1):
            for j in range(i+1, len(cand)):
                key = f'{cand[i]}_{cand[j]}'
                if key in pools.keys():
                    copy_pools[key] = pools[key]
                else:
                    key = f'{cand[j]}_{cand[i]}'
                    copy_pools[key] = pools[key]
    if iters < max_iters:
        val = [*copy_pools.values()][0]
        if val[0]>=val[1]:
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