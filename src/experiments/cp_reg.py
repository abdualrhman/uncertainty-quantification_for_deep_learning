import pandas as pd
from tqdm import tqdm
import itertools
import torch.backends.cudnn as cudnn
import random
import torch
import numpy as np
from src.utils.cp_reg_utils import *
from src.utils.utils import *
from src.utils.utils import get_model
import os
import sys
from src.models.CQR import CQR_logits

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def make_table(df, alpha):
    def round_to_n(x, n): return np.round(
        x, -int(np.floor(np.log10(x))) + (n - 1))  # Rounds to sig figs
    df = df[df.alpha == alpha]
    table = ""
    table += "\\begin{table}[t] \n"
    table += "\\centering \n"
    table += "\\small \n"
    table += "\\begin{tabular}{lccccccccccc} \n"
    table += "\\toprule \n"
    table += " & \multicolumn{2}{c}{Length}  & \multicolumn{2}{c}{Coverage}  \\\\ \n"
    table += "\cmidrule(r){2-3}  \cmidrule(r){4-5} \n"
    table += "Model & QR & CQR & QR & CQR \\\\ \n"
    table += "\\midrule \n"
    for model in df.Model.unique():
        df_model = df[df.Model == model]
        table += f" {model} & "
        table += str(round_to_n(
            df_model.Length[df_model.Predictor == "QR"].item(), 3)) + " & "
        table += str(round_to_n(
            df_model.Length[df_model.Predictor == "CQR"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Coverage"]
                     [df_model.Predictor == "QR"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Coverage"]
                     [df_model.Predictor == "CQR"].item(), 3)) + " \\\\ \n"

    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\caption{\\textbf{Results on California Housing Testset.} We report the prediction interval length and the coverage fot $\\alpha=0.1$ for two quantile regression models and for the same model conformalized} \n"
    table += "\\label{table:California_test_0.1} \n"
    table += "\\end{table} \n"
    return table


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


def experiment(modelname, datasetname, datasetpath, num_trials, alpha, n_data_conf, bsz, predictor):
    # Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)
    # Instantiate and wrap model
    model = get_model(modelname)
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
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cache_fname = "./.cache/cal_housing.csv"
    alpha_table = 0.1
    try:
        df = pd.read_csv(cache_fname)
    except:
        modelnames = ['GBQuantileReg', 'RegFFN']
        alphas = [0.1]
        predictors = ['QR', 'CQR']
        params = list(itertools.product(modelnames, alphas, predictors))
        m = len(params)
        datasetname = 'CalHousing'
        datasetpath = '/scratch/group/ilsvrc/val/'
        num_trials = 100
        kreg = None
        lamda = None
        randomized = True
        n_data_conf = 1000
        bsz = 32
        cudnn.benchmark = True

        # Perform the experiment
        df = pd.DataFrame(
            columns=["Model", "Predictor", "Length", "alpha", "Coverage"])
        for i in range(m):
            modelname, alpha, predictor = params[i]
            print(
                f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

            out = experiment(modelname, datasetname, datasetpath, num_trials,
                             params[i][1], n_data_conf, bsz, predictor)
            df = df.append({"Model": modelname,
                            "Predictor": predictor,
                            "Length": np.round(out[0], 3),
                            "Coverage": np.round(out[1], 3),
                            "alpha": alpha}, ignore_index=True)
        df.to_csv(cache_fname)
    # Print the TeX table
    table_str = make_table(df, alpha_table)
    table = open(
        f"outputs/california_housing_results_{alpha_table}".replace('.', '_') + ".tex", 'w')
    table.write(table_str)
    table.close()
