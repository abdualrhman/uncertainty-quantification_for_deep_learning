from src.models.conformal_model import *
import pandas as pd
from tqdm import tqdm
import itertools
import torch.backends.cudnn as cudnn
import random
import torchvision.transforms as tf
import torchvision
import torch.utils.data as tdata
import torch
# from scipy.stats import median_absolute_deviation as mad
from scipy.special import softmax
import numpy as np
from src.utils.cp_utils import *
import os
import sys
import inspect
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
    table += " & \multicolumn{3}{c}{Accuracy}  & \multicolumn{4}{c}{Coverage} & \multicolumn{4}{c}{Size} \\\\ \n"
    table += "\cmidrule(r){3-4}  \cmidrule(r){5-8}  \cmidrule(r){9-12} \n"
    table += "Model & F1 & Top-1 & Top-5 & Top K & Naive & APS & RAPS & Top K & Naive & APS & RAPS \\\\ \n"
    table += "\\midrule \n"
    for model in df.Model.unique():
        df_model = df[df.Model == model]
        table += f" {model} & "
        table += f" {np.round(df_model.F1.mean(), 3)} & "
        table += f" {np.round(df_model.Top1.mean(), 3)} & "
        table += f" {np.round(df_model.Top5.mean(), 3)} & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "Fixed"].item(), 3)) + " & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "RAPS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "Fixed"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "RAPS"].item(), 3)) + " \\\\ \n"

    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\caption{\\textbf{Results on Imagenet-Val.} We report coverage and size of the optimal, randomized fixed sets, \\naive, \\aps,\ and \\raps\ sets for nine different Imagenet classifiers. The median-of-means for each column is reported over 100 different trials at the 10\% level. See Section~\\ref{subsec:imagenet-val} for full details.} \n"
    table += "\\label{table:imagenet-val} \n"
    table += "\\end{table} \n"
    return table


def trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, naive_bool):
    logits_cal, logits_val = split2(logits, n_data_conf, len(
        logits)-n_data_conf)  # A new random split for every trial
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(
        logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        logits_val, batch_size=bsz, shuffle=False, pin_memory=True)

    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda, randomized=randomized,
                                           allow_zero_sets=True, naive=naive_bool, batch_size=bsz, lamda_criterion='size')
    # Collect results
    top1_avg, top5_avg, f1score_avg, cvg_avg, sz_avg = validate(
        loader_val, conformal_model, print_bool=False)
    return top1_avg, top5_avg, f1score_avg, cvg_avg, sz_avg


def experiment(modelname, datasetname, datasetpath, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor):
    # Experiment logic
    naive_bool = predictor == 'Naive'
    # fixed_bool = predictor == 'Fixed'
    if predictor in ['Fixed', 'Naive', 'APS']:
        kreg = 1
        lamda = 0  # No regularization.

    # Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    # Instantiate and wrap model
    model = get_model(modelname)

    # Perform experiment
    top1s = np.zeros((num_trials,))
    top5s = np.zeros((num_trials,))
    f1scores = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    sizes = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, f1score_avg, cvg_avg, sz_avg = trial(
            model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, naive_bool)
        top1s[i] = top1_avg
        top5s[i] = top5_avg
        f1scores[i] = f1score_avg
        coverages[i] = cvg_avg
        sizes[i] = sz_avg
        print(
            f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f},  F1: {np.median(f1scores[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(top1s), np.median(top5s), np.median(f1scores), np.median(coverages), np.median(sizes)


if __name__ == "__main__":
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cache_fname = "./.cache/cifar10_df.csv"
    alpha_table = 0.1
    try:
        df = pd.read_csv(cache_fname)
    except:
        # Configure experiment
        # modelnames = ['Cifar10ConvModel','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
        modelnames = ['Cifar10ConvModel', 'Cifar10Resnet20']
        alphas = [0.05, 0.10]
        predictors = ['Fixed', 'Naive', 'APS', 'RAPS']
        params = list(itertools.product(modelnames, alphas, predictors))
        m = len(params)
        datasetname = 'Cifar10'
        datasetpath = '/scratch/group/ilsvrc/val/'
        num_trials = 1
        kreg = None
        lamda = None
        randomized = True
        n_data_conf = 5000
        n_data_val = 5000
        pct_paramtune = 0.33
        bsz = 32
        cudnn.benchmark = True

        # Perform the experiment
        df = pd.DataFrame(
            columns=["Model", "Predictor", "Top1", "Top5", "alpha", "Coverage", "Size"])
        for i in range(m):
            modelname, alpha, predictor = params[i]
            print(
                f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

            out = experiment(modelname, datasetname, datasetpath, num_trials,
                             params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor)
            df = df.append({"Model": modelname,
                            "Predictor": predictor,
                            "Top1": np.round(out[0], 3),
                            "Top5": np.round(out[1], 3),
                            "F1": np.round(out[2], 3),
                            "alpha": alpha,
                            "Coverage": np.round(out[3], 3),
                            "Size":
                            np.round(out[4], 3)}, ignore_index=True)
        df.to_csv(cache_fname)
    # Print the TeX table
    table_str = make_table(df, alpha_table)
    table = open(
        f"outputs/cifar10results_{alpha_table}".replace('.', '_') + ".tex", 'w')
    table.write(table_str)
    table.close()
