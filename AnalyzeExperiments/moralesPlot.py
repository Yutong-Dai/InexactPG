'''
File: moralesPlot.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-12-20 14:45
Last Modified: 2020-12-23 15:12
--------------------------------------------
Description:
'''
from matplotlib import rc
rc('text', usetex=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plydata import *


class MoralesProfile:
    def __init__(self, algo_list, file_list, header='infer', align='datasetid'):
        self.algo_lst = algo_list
        self.file_lst = file_list
        # read file
        self.num_algo = len(self.algo_lst)
        if self.num_algo != len(self.file_lst):
            raise ValueError('Number of files not equal to the number of algorithm names provided.\n')
        self.df_lst = [] * len(self.file_lst)
        for file in self.file_lst:
            df = pd.read_table(file, header=header, delim_whitespace=True)
            if align is not None:
                df = df.sort_values(align).reset_index(drop=True)
            self.df_lst.append(df)
        # Confirm that the same number of lines has been read from all files
        base = self.df_lst[0].shape[0]
        for i in range(1, self.num_algo):
            if self.df_lst[i].shape[0] != base:
                err = 'Number of lines read from input file     : {}'.format(self.file_lst[0])
                err += 'differs from number read from input file : {}'.format(self.file_lst[i])
                raise ValueError(err)

    def plot(self, column, options={}, save=False, saveDir='./', ext=None, dpi=300, show_num=False, use_tt=True):
        if 'color' not in options.keys():
            options['color'] = 'rgb'
        if 'ratio_max' not in options.keys():
            options['ratio_max'] = 1
        self.options = options
        figure_lst = []
        for algo1 in range(self.num_algo - 1):
            for algo2 in range(algo1 + 1, self.num_algo):
                data1 = self.df_lst[algo1][column].copy()
                data2 = self.df_lst[algo2][column].copy()
                data1.loc[data1 < 0] = np.inf
                data2.loc[data2 < 0] = np.inf
                ratio = (data1 / data2).to_numpy()
                ratio[ratio == 0] = 1e-16
                ratio = -np.log2(ratio)
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
                figure = plt.figure()
                win = sum(flag)
                lose = sum(ratio < 0)
                if show_num:
                    label_1 = self.algo_lst[algo1] + ' {:3d}'.format(win)
                    label_2 = self.algo_lst[algo2] + ' {:3d}'.format(lose)
                else:
                    label_1 = self.algo_lst[algo1]
                    label_2 = self.algo_lst[algo2]
                    print('{} Win:{} | Lose:{}'.format(self.algo_lst[algo1], win, lose))
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
                else:
                    text = column
                plt.title("Metric: {}".format(text))
                figure_lst.append(figure)
                if save:
                    if ext is None:
                        filename = saveDir + '{}-{}_{}.eps'.format(self.algo_lst[algo1], self.algo_lst[algo2], column)
                    else:
                        filename = saveDir + '{}-{}_{}_{}.eps'.format(self.algo_lst[algo1], self.algo_lst[algo2], column, ext)
                    # metrics
                    if dpi:
                        figure.savefig(filename, dpi=dpi)
                    else:
                        figure.savefig(filename)
        return figure_lst

    def plot_facet(self, column, options={}, save=False, saveDir='./', ext=None):
        if 'color' not in options.keys():
            options['color'] = 'rgb'
        if 'ratio_max' not in options.keys():
            options['ratio_max'] = 1
        self.options = options
        for algo1 in range(self.num_algo - 1):
            for algo2 in range(algo1 + 1, self.num_algo):
                fig, axs = plt.subplots(2, 2, figsize=(10, 10), facecolor='w', edgecolor='k')
                axs = axs.ravel()
                count = 0
                for percent in [0.25, 0.50, 0.75, 1.00]:
                    df1 = self.df_lst[algo1]
                    df2 = self.df_lst[algo2]
                    data1 = (df1 >> query('percent == {}'.format(percent)))[column]
                    data2 = (df2 >> query('percent == {}'.format(percent)))[column]
                    ratio = (data1 / data2).to_numpy()
                    ratio = -np.log2(ratio)
                    bars_pos = np.zeros(ratio.shape)
                    bars_neg = np.zeros(ratio.shape)
                    flag = ratio > 0
                    bars_pos[flag] = ratio[flag]
                    bars_neg[~flag] = ratio[~flag]
                    bars_pos[::-1].sort()
                    bars_neg[::-1].sort()
                    win = sum(flag)
                    lose = sum(ratio < 0)
                    label_1 = self.algo_lst[algo1] + ' {:3d}'.format(win)
                    label_2 = self.algo_lst[algo2] + ' {:3d}'.format(lose)
                    if self.options['color'] == 'bw':
                        axs[count].bar(range(len(bars_pos)), bars_pos, color=[32 / 225, 32 / 225, 32 / 225, 1], label=label_1, width=1)
                        axs[count].bar(range(len(bars_neg)), bars_neg, color=[192 / 225, 192 / 225, 192 / 225, 1], label=label_2, width=1)
                    else:
                        axs[count].bar(range(len(bars_pos)), bars_pos, color='b', label=label_1, width=1)
                        axs[count].bar(range(len(bars_neg)), bars_neg, color='r', label=label_2, width=1)
                    axs[count].set_xlim(-1, len(bars_pos) + 1)
                    axs[count].set_ylim(-self.options['ratio_max'], self.options['ratio_max'])
                    axs[count].legend()
                    axs[count].hlines(y=0, xmin=0 - 0.5, xmax=len(bars_pos) - 0.5, linewidth=1)
                    axs[count].set_ylabel('-log2(ratio)')
                    axs[count].set_title('percent:{:3.2f}'.format(percent))
                    count += 1
                if save:
                    if ext is None:
                        filename = saveDir + './{}-{}.png'.format(self.algo_lst[algo1], self.algo_lst[algo2])
                    else:
                        filename = saveDir + './{}-{}_{}.png'.format(self.algo_lst[algo2], self.algo_lst[algo2], ext)
                    fig.savefig(filename)
