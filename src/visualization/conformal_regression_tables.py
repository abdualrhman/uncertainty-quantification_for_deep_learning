import itertools
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
    table += "Model & Avg. length & Coverage & $\\alpha$ \\\\ \n"
    table += "\\midrule \n"
    for row in df.itertuples():
        table += f" {row.Predictor} {row.Model} & "
        table += f"{round_to_n(row.Length,3 )} & "
        table += f"{round_to_n(row.Coverage, 3)} & "
        table += f"{row.alpha} " + " \\\\ \n"
    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\end{table} \n"
    return table


if __name__ == "__main__":
    datasets = ['california_housing', 'wine_quality']
    for datasetname in datasets:
        try:
            cache_fname = f".cache/conformal_regression_{datasetname}.csv"
            df = pd.read_csv(cache_fname)
            CQR_table_str = make_table(df[df.Predictor == 'CQR'])
            table = open(
                f"reports/tables/CQR_{datasetname}_results.tex", 'w')
            table.write(CQR_table_str)
            table.close()
            QR_table_str = make_table(df[df.Predictor == 'QR'])
            table = open(
                f"reports/tables/QR_{datasetname}_results.tex", 'w')
            table.write(QR_table_str)
            table.close()
        except:
            print(
                f"No cashed result found for {datasetname}!")
            print(f"Refer to README.md to run the experiment and produce the results")
