from cProfile import label
import itertools
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


sys.path.insert(1, os.path.join(sys.path[0], '..'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="Cifar10")
    args = parser.parse_args()
    datasetname = args.dataset
    dirpath = './.cache/active_learning_experiments'
    sample_size = 1000
    model_default_acc = 0.702 if datasetname == 'Cifar10' else 0.552
    modelnames = ['Cifar10ConvModel']
    strategies = [
        'least-confidence',
        # 'conformal-score:least-confidence',
        'conformal-score:largest-set',
        'random-sampler',
        'entropy-sampler'
    ]
    params = list(itertools.product(strategies, modelnames))

    fig = plt.figure()
    for strategy, modelname in params:
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
    plt.axhline(y=model_default_acc, color='purple',
                linestyle='--', label='model accuracy')
    plt.legend()
    plt.ylabel('Model accuracy')
    plt.xlabel('Round')
    plt.title("Active learning round accuracies")
    plt.grid()
    plt.savefig(f'reports/figures/active_learning_{datasetname}_{sample_size}')
