
import argparse
import pandas as pd
import numpy as np
import torch
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class AMZN_SP(Dataset):
    def __init__(self, train: bool = True, in_folder: str = "data/raw", out_folder: str = "data/processed") -> None:
        super().__init__()
        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.fname = 'AMZN.csv'  # if train else 'AMZNtest.csv'
        self.fpath = f'{self.in_folder}/{self.fname}'

        if self.out_folder:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                print("Processed data")

        content = self.read_raw()
        content = content[['Date', 'Close']]
        content['Date'] = pd.to_datetime(content['Date'])
        lookback = 7
        shifted_df = self.prepare_dataframe_for_lstm(content, lookback)

        shifted_df_as_np = shifted_df.to_numpy()
        # transform
        scaler = MinMaxScaler(feature_range=(-1, 1))
        shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

        X = shifted_df_as_np[:, 1:]
        y = shifted_df_as_np[:, 0]

        X = dc(np.flip(X, axis=1))
        split_index = int(len(X) * 0.2)
        if self.train:
            X = X[split_index:]
            y = y[split_index:]
        else:
            X = X[:split_index]
            y = y[:split_index]

        self.data = torch.tensor(X.reshape((-1, lookback, 1))).float()
        self.targets = torch.tensor(y).float()

        self.save_preprocessed()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def load_preprocessed(self) -> None:

        fname = f"{self.out_folder}/AMZNtrain_processed.pt" if self.train else f"{self.out_folder}/AMZNtest_processed.pt"
        try:
            self.data, self.targets = torch.load(fname)
        except:
            raise ValueError("No preprocessed files found")

    def read_raw(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.fpath)
        except:
            raise ValueError(
                f'No data found at {self.fpath} \n Download from https://www.kaggle.com/datasets/prasoonkottarathil/amazon-stock-price-20142019')

    def save_preprocessed(self) -> None:

        fname = f"{self.out_folder}/AMZNtrain_processed.pt" if self.train else f"{self.out_folder}/AMZNtest_processed.pt"
        torch.save([self.data, self.targets], fname)

    def prepare_dataframe_for_lstm(self, df, n_steps):
        df = dc(df)

        df.set_index('Date', inplace=True)

        for i in range(1, n_steps+1):
            df[f'Close(t-{i})'] = df['Close'].shift(i)

        df.dropna(inplace=True)

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data object arguments")
    parser.add_argument("--in_folder", default="data/raw")
    parser.add_argument("--out_folder", default="data/processed")
    args = parser.parse_args()
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    train_set = AMZN_SP(train=True, in_folder=args.in_folder,
                        out_folder=args.out_folder)
    train_set.save_preprocessed()
    test_set = AMZN_SP(train=False, in_folder=args.in_folder,
                       out_folder=args.out_folder)
    test_set.save_preprocessed()
