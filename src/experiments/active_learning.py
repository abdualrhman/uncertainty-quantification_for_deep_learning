import argparse
import pandas as pd
import torch.backends.cudnn as cudnn
import random
import torch
import numpy as np
import os
import sys
from src.utils.cp_utils import *
from src.utils.utils import *
from src.models.oracle import Oracle
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def get_teaching_round_num(sample_size, trainset_size):
    return 1


def trail(modelname, train_set, test_set, strategy, sample_size, n_init_training_labels, n_training_epochs, calib_size, alpha):

    train_set = pre_transform_dataset(train_set)
    test_set = pre_transform_dataset(test_set)
    model = get_untrained_model(modelname)

    oracle = Oracle(model=model, train_set=train_set, test_set=test_set, strategy=strategy, sample_size=sample_size,
                    n_init_training_labels=n_init_training_labels, n_training_epochs=n_training_epochs, calib_size=calib_size, alpha=alpha)
    oracle.teach(40)

    return oracle.round_accuracies
    # return np.random.randint(low=0, high=100, size=40)


def experiment(modelname, train_set, test_set, n_trials, strategy, sample_size, n_init_training_labels, n_training_epochs, calib_size, alpha):
    trails_rounds_accuracies = []
    for i in range(n_trials):
        print(f"trail {i+1}/{n_trials}")
        round_accuracies = trail(modelname, train_set, test_set, strategy, sample_size, n_init_training_labels, n_training_epochs, calib_size,
                                 alpha)
        trails_rounds_accuracies.append(round_accuracies)
    return trails_rounds_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--strategy", default="least-confidence")
    parser.add_argument("--model", default="Cifar10ConvModel")
    parser.add_argument("--sample_size", default=1000)
    args = parser.parse_args()
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    datasetname = args.dataset
    strategy = args.strategy
    modelname = args.model
    sample_size = int(args.sample_size)
    dirpath = './.cache/active_learning_experiments'
    cache_fname = f"{dirpath}/{datasetname}_{strategy}_{modelname}_{sample_size}.csv"
    try:
        df = pd.read_csv(cache_fname)
    except:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        n_trials = 5
        n_init_training_labels = 1000
        n_training_epochs = 1
        alpha = 0.05
        calib_size = 500
        bsz = 32
        train_set = get_dataset(datasetname, train=True)
        test_set = get_dataset(datasetname, train=False)
        cudnn.benchmark = True
        # Perform the experiment
        df = pd.DataFrame(
            columns=["dataset", "model", "strategy", "round_accuries"])
        trails_rounds_accuracies = experiment(
            modelname, train_set, test_set, n_trials, strategy, sample_size, n_init_training_labels, n_training_epochs, calib_size, alpha)
        df = df.append({"dataset": datasetname,
                        "model": modelname,
                        "strategy": strategy,
                        "round_accuries": trails_rounds_accuracies
                        },
                       ignore_index=True)
        df.to_csv(cache_fname)

    # Print the TeX table
    # table_str = make_table(df, alpha_table)
    # table = open(
        # f"outputs/active_learning_CIFAR10_results".replace('.', '_') + ".tex", 'w')
    # table.write(table_str)
    # table.close()
