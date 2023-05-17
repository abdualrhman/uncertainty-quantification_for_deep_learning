import itertools
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))

if __name__ == "__main__":
    dirpath = './.cache/active_learning_experiments'
    sample_size = [1000]
    datasetname = ['Cifar10']
    modelnames = ['Cifar10ConvModel']
    strategies = ['least-confidence',
                  'conformal-score:least-confidence', 'random-sampler', 'entropy-sampler']
    params = list(itertools.product(
        datasetname, strategies, modelnames, sample_size))

    fig = plt.figure(figsize=(8, 11))
    for datasetname, strategy, modelname, sample_size in params:
        cache_fname = f"{dirpath}/{datasetname}_{strategy}_{modelname}_{sample_size}.csv"
        df = pd.read_csv(cache_fname)
        str_arr = df.round_accuries.values
        # convert string to np array
        string = str_arr[0].replace("array(", "").replace("dtype=object)", "")
        trail_acc = np.array(eval(string))

        std = trail_acc.std(axis=0)
        avg = trail_acc.mean(axis=0)
        if strategy == 'random-sampler':
            plt_color = 'green'
        elif strategy == 'least-confidence':
            plt_color = 'blue'
        elif strategy == 'entropy-sampler':
            plt_color = 'black'
        else:
            plt_color = 'red'
        plt.plot(avg, linestyle='-', color=plt_color,
                 label=f"{strategy.replace('-', ' ')} mean")
        plt.plot(avg+std, ':', color=plt_color,
                 label=f"{strategy.replace('-', ' ')} ± std")
        plt.plot(avg-std, ':', color=plt_color)
        plt.legend()
        plt.grid()
    plt.savefig(f'reports/figures/active_learning_CIFAR10_{sample_size}')
