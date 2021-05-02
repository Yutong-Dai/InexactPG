import sys
import glob
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


def create_paths(logdir='../IPG/test/log', date='04_18_2021', solver='naive', loss='logit', lam_shrink=[0.1, 0.01], percent=[0.1, 0.2], excludes=None, *argv):
    list_all_npy_path =[]
    for lam in lam_shrink:
        for per in percent:
            minimal_dir = f'{logdir}/{date}/{solver}/{loss}/{lam}_{per}'
            for arg in argv:
                minimal_dir += f'_{arg}'
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

def get_all_naive(logdir, date, loss, lam_shrink, percent, safeguard, safeguard_consts=[1e-1, 1e0, 1e1], ts=[1e-12, 1e-3, 1e0], excludes=None):
    algo_df_dict = {}
    solver = 'naive'
    if safeguard == 'const':
        safeguard_const = 1e3
        for t in ts:
            algorithm = f'{solver}-{t}-{safeguard}-{safeguard_const}'
            print(f'{algorithm}')
            paths = create_paths(logdir, date, solver, loss, lam_shrink, percent, excludes, t, 'const', 1e3)
            if paths == []:
                print(' empty')
                df = None
            else:
                df = load_df_from_paths(paths)
            algo_df_dict[algorithm] = df
    else:
        for safeguard_const in safeguard_consts:
            for t in ts:
                algorithm = f'{solver}-{t}-{safeguard}-{safeguard_const}'
                print(f'{algorithm}')
                paths = create_paths(logdir, date, solver, loss, lam_shrink, percent, excludes, t, safeguard, safeguard_const)
                if paths == []:
                    print(' empty')
                    df = None
                else:
                    df = load_df_from_paths(paths)
                algo_df_dict[algorithm] = df
    return algo_df_dict

def get_all_schimdt(logdir, date, loss, lam_shrink, percent, schimdt_consts=[1e-1, 1e0, 1e1], excludes=None):
    solver = 'schimdt'
    algo_df_dict = {}
    for schimdt_const in schimdt_consts:
        algorithm = f"{solver}-{'none'}-{schimdt_const}"
        print(f'{algorithm}')
        paths = create_paths(logdir, date, 'schimdt', loss, lam_shrink, percent, excludes, "'none'", schimdt_const)
        if paths == []:
            print(' empty')
            df = None
        else:
            df = load_df_from_paths(paths)
        algo_df_dict[algorithm] = df
    return algo_df_dict

def get_all_adaptive(logdir, date, loss, lam_shrink, percent, ts=[1e-12], excludes=None, solver = 'naive'):
    algo_df_dict = {}
    for t in ts:
        algorithm = f'{solver}-{t}-none-inf'
        print(f'{algorithm}')
        paths = create_paths(logdir, date, solver, loss, lam_shrink, percent, excludes, t, 'none', 'inf')
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
        print(f"After subsetting, {num_records} instances are kept.")
    def plot(self, column, options={}, save=False, saveDir='./', ext=None, dpi=300, show_num=False, use_tt=True, plot=True, aoc=True, factor=1.5):
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
                ratio[ratio == -np.log2(1e-16)] = np.inf
                ratio = ratio[~np.isnan(ratio)]
                bars_pos = np.zeros(ratio.shape)
                bars_neg = np.zeros(ratio.shape)
                flag = ratio > 0
                bars_pos[flag] = ratio[flag]
                bars_neg[~flag] = ratio[~flag]
                if aoc:
                    posmax = np.max(bars_pos[bars_pos != np.inf])
                    negmin = np.min(bars_neg[bars_neg != -np.inf])
                    ratio_max = max(posmax, -negmin)
                    bars_pos[bars_pos == np.inf] = ratio_max * factor
                    bars_neg[bars_neg == -np.inf] = ratio_max * factor
                    self.bars_pos = bars_pos
                    self.bars_neg = bars_neg
                else:
                    bars_pos[bars_pos == np.inf] = self.options['ratio_max']
                    bars_neg[bars_neg == -np.inf] = -self.options['ratio_max']
                # sort in descending order
                bars_pos[::-1].sort()
                bars_neg[::-1].sort()
                if aoc:
                    win_aoc =  np.around(np.sum(bars_pos), 3)
                    lose_aoc = np.around(np.abs(np.sum(bars_neg)), 3)
                win = sum(flag)
                lose = sum(ratio < 0)
                if plot:
                    figure = plt.figure()
                    if show_num:
                        label_1 = algo1 + ' {}'.format(win)
                        label_2 = algo2 + ' {}'.format(lose)
                        if aoc:
                            label_1 += f' aoc:{win_aoc}'
                            label_2 += f' aoc:{lose_aoc}'
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
                    if aoc:
                        plt.title("Metric: {} (Area Under the Curve)".format(text))
                    else:
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
                    if aoc:
                        pools[key] = (win_aoc, lose_aoc, algo1, algo2)
                    else:
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
    

        
def get_one_instance(date, loss, dataset, lambda_shrinage, percent, params_naive_const, 
                     params_naive_laststep, params_naive_schimdt, params_schimdt):
    label_lst = []
    Fseq_lst = []
    Eseq_lst = []
    Gseq_lst = []
    subgrad_lst = []
    time_lst = []
    naive_dir_base = f'../IPG/test/log/{date}/naive/{loss}/{lambda_shrinage}_{percent}'
    naive_const_dir = f'{naive_dir_base}_{params_naive_const[0]}_const_{params_naive_const[1]}/{dataset}_info.npy'
    naive_laststep_dir = f'{naive_dir_base}_{params_naive_laststep[0]}_laststep_{params_naive_laststep[1]}/{dataset}_info.npy'
    naive_schimdt_dir = f'{naive_dir_base}_{params_naive_schimdt[0]}_schimdt_{params_naive_schimdt[1]}/{dataset}_info.npy'    
    schimdt_dir = f"../IPG/test/log/{date}/schimdt/{loss}/{lambda_shrinage}_{percent}_'none'_{params_schimdt[1]}/{dataset}_info.npy"
    dir_lst = [naive_const_dir, naive_laststep_dir, naive_schimdt_dir, schimdt_dir]
    label_lst = [f'const-{params_naive_const[0]}-{params_naive_const[1]}',
                 f'laststep-{params_naive_laststep[0]}-{params_naive_laststep[1]}',
                 f'schimdt-{params_naive_schimdt[0]}-{params_naive_schimdt[1]}',
                 f'Schimdt-{params_schimdt[1]}',
                ]
    label_lst = [f'{i:20}' for i in label_lst]
    for i in dir_lst:
        info = np.load(i, allow_pickle=True).item()
        Fseq_lst.append(info['Fseq'])
        Eseq_lst.append(info['Eseq'])
        Gseq_lst.append(info['Gseq'])
        subgrad_lst.append(info['subgrad_iters'])
        time_lst.append(info['time'])
    return Fseq_lst, Eseq_lst, Gseq_lst, subgrad_lst, time_lst, label_lst

def create_plot(dataset, lambda_shrinage, percent, Fseq_lst, Eseq_lst, 
                Gseq_lst, subgrad_lst, time_lst, label_lst, 
                fxmax, exmax, alpha, markerset='fill', savedir=None, ext=None):
    fig, (row1, row2, row3) = plt.subplots(1, 3, figsize=(18,4))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if markerset == 'fill':
        markers=['o', 's', 'P', '*', 'D', 'X', 'p']
    else:
        markers=['.', '1', 'x', '4', '+', '|', '_']        
    for i in range(len(label_lst)):
        row1.plot(Fseq_lst[i], markers[i], color=colors[i], label=f'{label_lst[i]}: iters:{len(Fseq_lst[i])}', alpha=alpha)
    for i in range(len(label_lst)):
        row2.plot(Eseq_lst[i], markers[i], color=colors[i], label=f'{label_lst[i]}: subgrad:{subgrad_lst[i]:2.2e}', alpha=alpha)
    for i in range(len(label_lst)):
        row3.plot(Gseq_lst[i], markers[i], color=colors[i], label=f'{label_lst[i]}: time:{time_lst[i]:2.2e}', alpha=alpha) 
    row1.set_title("Fseq")
    row2.set_title("Eseq")
    row3.set_title("Gseq")    
    row1.legend()
    row2.legend()
    row3.legend()
    if fxmax != -1:
        row1.set_xlim((-1, fxmax))
    row1.set_yscale('log')
    if exmax != -1:
        row2.set_xlim((-1, exmax))
        row3.set_xlim((-1, exmax))
    row2.set_yscale('log')
    row3.set_yscale('log')
    dname = " ".join(dataset.split('_'))
    fig.suptitle(f'{dname}-{lambda_shrinage}-{percent}', fontsize="x-large",  y=1.02)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.0)
    if savedir is not None:
        if ext is not None:
            plt.savefig(f'{savedir}/{dataset}_{lambda_shrinage}_{percent}_{ext}.png', bbox_inches = "tight")
        else:
            plt.savefig(f'{savedir}/{dataset}_{lambda_shrinage}_{percent}.png', bbox_inches = "tight")
        plt.close(fig)