
import argparse
import pandas as pd
import numpy as np
import torch

from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class WineQuality(Dataset):
    def __init__(self, train: bool = True, in_folder: str = "data/raw", out_folder: str = "data/processed") -> None:
        super().__init__()
        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.fname = 'winequality-red.csv'
        self.fpath = f'{self.in_folder}/{self.fname}'

        if self.out_folder:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                print("Saved data files not found!")
        print("Processing new data ...")
        content = self.read_raw()
        input_cols = ['fixed acidity',
                      'volatile acidity',
                      'citric acid',
                      'residual sugar',
                      'chlorides',
                      'free sulfur dioxide',
                      'total sulfur dioxide',
                      'density',
                      'pH',
                      'sulphates',
                      'alcohol']
        traget_cols = ['quality']
        X = content[input_cols].to_numpy()
        y = content[traget_cols].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        if self.train:
            self.data = torch.tensor(X_train).float()
            self.targets = torch.tensor(y_train).view(-1).float()
        else:
            self.data = torch.tensor(X_test).float()
            self.targets = torch.tensor(y_test).view(-1).float()

        self.save_preprocessed()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def load_preprocessed(self) -> None:

        fname = f"{self.out_folder}/wine_quality_train_processed.pt" if self.train else f"{self.out_folder}/wine_quality_test_processed.pt"
        try:
            self.data, self.targets = torch.load(fname)
        except:
            raise ValueError("No preprocessed files found")

    def read_raw(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.fpath)
        except:
            raise ValueError(
                f'No data found at {self.fpath} \n Download from https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?datasetId=4458')

    def save_preprocessed(self) -> None:

        fname = f"{self.out_folder}/wine_quality_train_processed.pt" if self.train else f"{self.out_folder}/wine_quality_test_processed.pt"
        torch.save([self.data, self.targets], fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data object arguments")
    parser.add_argument("--in_folder", default="data/raw")
    parser.add_argument("--out_folder", default="data/processed")
    args = parser.parse_args()
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    train_set = WineQuality(train=True, in_folder=args.in_folder,
                            out_folder=args.out_folder)
    train_set.save_preprocessed()
    test_set = WineQuality(train=False, in_folder=args.in_folder,
                           out_folder=args.out_folder)
    test_set.save_preprocessed()
