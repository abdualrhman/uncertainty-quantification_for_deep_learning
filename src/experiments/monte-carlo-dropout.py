import argparse
import warnings
import pandas as pd
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import numpy as np
from src.utils.cp_reg_utils import *
from src.utils.utils import *
import os
import sys


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", UserWarning)

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def trial(model, val_loader, alpha, n_forward_passes, n_classes, n_samples):
    # get (1-alpha)% prediction sets
    _, S = get_monte_carlo_prediction_sets(
        model=model, val_loader=val_loader, forward_passes=n_forward_passes, n_classes=n_classes, n_samples=n_samples, alpha=alpha)
    coverage_avg, size_avg, f1_avg = validate_monte_carlo_prediction(
        val_loader=val_loader, predictions_sets=S)
    return coverage_avg, size_avg, f1_avg


def experiment(modelname, datasetname, num_trials, n_forward_passes, alpha):
    # Data Loading
    val_set = get_dataset(datasetname, train=False)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    # Instantiate and wrap model
    model = get_model(modelname, datasetname)
    # Perform experiment
    sizes = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    f1_scores = np.zeros((num_trials,))
    n_classes = np.unique(val_set.targets).max()+1
    n_samples = len(val_set.targets)
    for i in tqdm(range(num_trials)):
        cvg_avg, sze_avg, f1_avg = trial(
            model=model, val_loader=val_loader, alpha=alpha, n_classes=n_classes, n_samples=n_samples, n_forward_passes=n_forward_passes)
        coverages[i] = cvg_avg
        sizes[i] = sze_avg
        f1_scores[i] = f1_avg
        print(
            f'\n\t Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}, F1_score {np.median(f1_scores[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(coverages), np.median(sizes),  np.median(f1_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--num_trials", default=50)
    parser.add_argument("--n_forward_passes", default=50)
    args = parser.parse_args()

    datasetname = args.dataset
    cache_fname = f".cache/classification_monte_carlo_{datasetname}.csv"

    try:
        df = pd.read_csv(cache_fname)
    except:
        modelnames = ['Cifar10ConvModel']
        alpha = 0.1
        n_forward_passes = int(args.n_forward_passes)
        num_trials = int(args.num_trials)
        cudnn.benchmark = True
        # Perform the experiment
        df = pd.DataFrame(
            columns=["Model", "Coverage", "Size", "F1"])
        for modelname in modelnames:

            print(
                f'Model: {modelname} | Desired coverage: {1-alpha}')

            out = experiment(modelname=modelname, datasetname=datasetname,
                             num_trials=num_trials, n_forward_passes=n_forward_passes, alpha=alpha)
            df = df.append({"Model": modelname,
                            "Coverage": np.round(out[0], 3),
                            "Size": np.round(out[1], 3),
                            "F1": np.round(out[2], 3),
                            }, ignore_index=True)
        df.to_csv(cache_fname)
