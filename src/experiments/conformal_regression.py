import argparse
import warnings
import pandas as pd
from tqdm import tqdm
import itertools
import torch.backends.cudnn as cudnn
import random
import torch
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


def trial(model, logits, alpha, n_data_conf, bsz, predictor):
    precomputed_logits = predictor == 'QR'
    logits_cal, logits_val = split2(logits, n_data_conf, len(
        logits)-n_data_conf)  # A new random split for every trial
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        logits_val, batch_size=bsz, shuffle=False, pin_memory=True)

    len_avg, cvg_avg = validate(
        loader_val, loader_cal, model, alpha=alpha, precomputed_logits=precomputed_logits, print_bool=False)

    return len_avg, cvg_avg


def experiment(modelname, datasetname, num_trials, alpha, n_data_conf, bsz, predictor):
    # Data Loading
    logits = get_logits_dataset(modelname, datasetname)
    # Instantiate and wrap model
    model = get_model(modelname, datasetname)
    # Perform experiment
    lengths = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        len_avg, cvg_avg = trial(
            model, logits, alpha, n_data_conf, bsz, predictor)
        lengths[i] = len_avg
        coverages[i] = cvg_avg
        print(
            f'\n\t Length: {np.median(lengths[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(lengths), np.median(coverages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="california_housing")
    args = parser.parse_args()
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    datasetname = args.dataset
    cache_fname = f".cache/conformal_regression_{datasetname}.csv"

    try:
        df = pd.read_csv(cache_fname)
    except:
        modelnames = ['GradientBoostingRegressor:0.05', 'GradientBoostingRegressor:0.1', 'GradientBoostingRegressor:0.2', 'CatBoostingRegressor:0.05', 'CatBoostingRegressor:0.1', 'CatBoostingRegressor:0.2', 'QuantileNet:0.05', 'QuantileNet:0.1',
                      'QuantileNet:0.2']
        predictors = ['QR', 'CQR']
        params = list(itertools.product(modelnames, predictors))
        m = len(params)

        num_trials = 100
        n_data_conf = 100 if datasetname == 'wine_quality' else 1000
        bsz = 32
        cudnn.benchmark = True
        # Perform the experiment
        df = pd.DataFrame(
            columns=["Model", "Predictor", "Length", "alpha", "Coverage"])
        for i in range(m):
            modelname,  predictor = params[i]
            alpha = get_model_alpha(modelname)
            print(
                f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

            out = experiment(modelname, datasetname, num_trials, alpha,
                             n_data_conf, bsz, predictor)
            df = df.append({"Model": modelname,
                            "Predictor": predictor,
                            "Length": np.round(out[0], 3),
                            "Coverage": np.round(out[1], 3),
                            "alpha": alpha}, ignore_index=True)
        df.to_csv(cache_fname)
