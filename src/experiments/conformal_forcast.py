import argparse
from enum import unique
import pandas as pd
import torch.backends.cudnn as cudnn
import random
import torch
import numpy as np
import os
import sys
from src.models.conformal_forcast import ConformalForcast
from src.utils.cp_utils import *
from src.utils.utils import *
from src.utils.conformal_forcaster_utils import validate_forcaster
from src.models.oracle import Oracle
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def make_table(df):
    def round_to_n(x, n): return np.round(
        x, -int(np.floor(np.log10(x))) + (n - 1))  # Rounds to sig figs
    alpha_set = {" ".join(f'{str(i)},' for i in df.alpha.unique())}
    table = ""
    table += "\\begin{table}[t] \n"
    table += "\\centering \n"
    table += "\\small \n"
    table += "\\begin{tabular}{lccccccccccc} \n"
    table += "\\toprule \n"
    table += "Model & Avg. length & Coverage & $ \\alpha$ \\\\ \n"
    table += "\\midrule \n"
    for model in df.Model.unique():
        for alpha in df.alpha.unique():
            df_model = df[df.Model == model]
            table += f"\\verb|{model}| & "
            table += str(round_to_n(
                df_model.Length[df_model.alpha == alpha].item(), 3)) + " & "
            table += str(round_to_n(df_model["Coverage"]
                         [df_model.alpha == alpha].item(), 3)) + " & "
            table += str(round_to_n(
                df_model.alpha[df_model.alpha == alpha].item(), 3)) + " \\\\ \n"

    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\caption{\\textbf{Results on Amazon Stock Price dataset.} We report the prediction interval length and the coverage for $\\alpha \in \{ 0.2, 0.1, 0.05 \}$ for a conformalized LSTM model.} \n"
    table += "\\label{table:amazon_stock_price.1} \n"
    table += "\\end{table} \n"
    return table


def trail(modelname, test_set, alpha, n_data_conf):
    model = get_model(modelname)
    cal_set, val_set = torch.utils.data.random_split(
        test_set, [n_data_conf, len(test_set)-n_data_conf])

    calib_loader = torch.utils.data.DataLoader(
        cal_set, batch_size=bsz, shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bsz, shuffle=False, pin_memory=True)

    conformal_model = ConformalForcast(
        model=model, calib_loader=calib_loader, alpha=alpha)
    len_avg, cvg_avg = validate_forcaster(
        val_loader, conformal_model, print_bool=False)

    return len_avg, cvg_avg


def experiment(modelname: str, test_set, n_trials: int,  alpha: float, n_data_conf: int):
    # Data Loading
    lengths = np.zeros((n_trials,))
    coverages = np.zeros((n_trials,))
    for i in tqdm(range(n_trials)):
        len_avg, cvg_avg = trail(modelname, test_set, alpha, n_data_conf)
        lengths[i] = len_avg
        coverages[i] = cvg_avg
        print(
            f'\n\t alpha:{alpha}, Length: {np.median(lengths[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}\033[F', end='')

    print('')
    return np.median(lengths), np.median(coverages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="AMZN")
    parser.add_argument("--model", default="LSTM_AMZN")
    args = parser.parse_args()
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    datasetname = args.dataset
    modelname = args.model
    dirpath = './.cache/conformal_forcast'
    cache_fname = f"{dirpath}/{datasetname}_{modelname}.csv"
    try:
        df = pd.read_csv(cache_fname)
    except:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        n_trials = 15
        alphas = [0.2, 0.1, 0.05]
        bsz = 16
        n_data_conf = 200
        test_set = get_dataset(datasetname, train=False)
        cudnn.benchmark = True
        # Perform the experiment
        df = pd.DataFrame(
            columns=["Model", "Length", "Coverage", "alpha"])
        for alpha in alphas:
            out = experiment(modelname, test_set,
                             n_trials,  alpha, n_data_conf)
            df = df.append({"Model": modelname,
                            "Length": np.round(out[0], 3),
                            "Coverage": np.round(out[1], 3),
                            "alpha": alpha
                            },
                           ignore_index=True)
        df.to_csv(cache_fname)
    table_str = make_table(df)
    table = open(
        f"reports/tables/{datasetname}_{modelname}_table".replace('.', '_') + ".tex", 'w')
    table.write(table_str)
    table.close()
