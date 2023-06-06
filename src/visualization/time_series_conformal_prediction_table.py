import argparse
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exeriment arguments")
    parser.add_argument("--dataset", default="AMZN")
    parser.add_argument("--model", default="LSTM_AMZN")
    args = parser.parse_args()
    modelname = args.model
    datasetname = args.dataset
    try:
        cache_fname = f".cache/time_series_conformal_prediction_{datasetname}_{modelname}.csv"
        df = pd.read_csv(cache_fname)
        table_str = make_table(df)
        table = open(
            f"reports/tables/time_series_conformal_prediction_{modelname}_table".replace('.', '_') + ".tex", 'w')
        table.write(table_str)
        table.close()
    except:
        print(
            f"No cashed result found for {datasetname}!")
        print(f"Refer to README.md to run the experiment and produce the results")
