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
    table += "\\begin{tabular}{lccc} \n"
    table += "\\toprule \n"
    table += "Model & Coverage & Size & F1 \\\\ \n"
    table += "\\midrule \n"
    for row in df.itertuples():
        table += f"{row.Model} & "
        table += f"{round_to_n(row.Coverage,3 )} & "
        table += f"{round_to_n(row.Size, 3)} & "
        table += f"{round_to_n(row.F1, 3)}  \\\\ \n"
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
        cache_fname = f".cache/classification_monte_carlo_{datasetname}.csv"
        df = pd.read_csv(cache_fname)
        CQR_table_str = make_table(df)
        table = open(
            f"reports/tables/classification-monte-carlo-table-{datasetname}.tex", 'w')
        table.write(CQR_table_str)
        table.close()
    except:
        print(
            f"No cashed result found for {datasetname}!")
        print(f"Refer to README.md to run the experiment and produce the results")
