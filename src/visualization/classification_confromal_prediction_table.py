import argparse
import numpy as np
import pandas as pd


def make_table(df):
    def round_to_n(x, n): return np.round(
        x, -int(np.floor(np.log10(x))) + (n - 1))  # Rounds to sig figs
    table = ""
    table += "\\begin{table}[t] \n"
    table += "\\centering \n"
    table += "\\small \n"
    table += "\\begin{tabular}{lccccccccccc} \n"
    table += "\\toprule \n"
    table += "  & \multicolumn{1}{c}{Accuracy}  & \multicolumn{3}{c}{Coverage} & \multicolumn{3}{c}{Size}  \\\\ \n"
    table += "\cmidrule(r){2-2}  \cmidrule(r){3-5}  \cmidrule(r){6-8}  \n"
    table += "Model & F1 & Naive & APS & RAPS & Naive & APS & RAPS  \\\\ \n"
    table += "\\midrule \n"
    for model in df.Model.unique():
        df_model = df[df.Model == model]
        table += f" {model} & "
        table += f" {np.round(df_model.F1.mean(), 3)} & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(
            df_model.Coverage[df_model.Predictor == "RAPS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"]
                     [df_model.Predictor == "RAPS"].item(), 3)) + " \\\\ \n"

    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\end{table} \n"
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="Cifar10")
    args = parser.parse_args()
    datasetname = args.dataset
    try:
        cache_fname = f".cache/classification_conformal_prediction_{datasetname}.csv"
        df = pd.read_csv(cache_fname)
        CQR_table_str = make_table(df)
        table = open(
            f"reports/tables/classification-conformal-prediction-table-{datasetname}.tex", 'w')
        table.write(CQR_table_str)
        table.close()
    except:
        print(
            f"No cashed result found for {datasetname}!")
        print(f"Refer to README.md to run the experiment and produce the results")
