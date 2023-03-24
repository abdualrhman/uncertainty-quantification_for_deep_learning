import logging
import click
import numpy as np
from sklearn.datasets import fetch_california_housing
import torch
from torch.utils.data import Dataset
import os
from torch import Tensor
from typing import Tuple


class CaliforniaHousing(Dataset):
    def __init__(self, split: str, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()

        if split not in ['train', 'calib', 'test']:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.in_folder = in_folder + '/cal_housing'
        self.out_folder = out_folder + '/cal_housing'

        if self.out_folder:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                pass
        self.download_data()

        content = np.load(f"{self.in_folder}/{self.split}_set.npy")
        self.data = torch.tensor(content[:, :-1])
        self.targets = torch.tensor(content[:, -1:].reshape(-1))

        if self.out_folder:
            self.save_preprocessed()

    def load_preprocessed(self) -> None:
        try:
            self.data, self.targets = torch.load(
                f"{self.out_folder}/{self.split}_processed.pt")
        except:
            raise ValueError("No preprocessed files found")

    def download_data(self) -> None:
        df = fetch_california_housing(
            data_home=self.in_folder, download_if_missing=True, as_frame=True)
        df.data['target'] = df.target
        data = df.data.to_numpy()
        train_num = int(0.75*data.shape[0])
        calib_num = int(0.05*data.shape[0])
        train_idx = train_num
        calib_idx = train_idx + calib_num
        test_idx = calib_idx + train_idx

        train_set = data[:train_idx]
        calib_set = data[train_idx:calib_idx]
        test_set = data[calib_idx:test_idx]

        assert train_set.shape[0]+calib_set.shape[0] + \
            test_set.shape[0] == data.shape[0]

        np.save(f"{self.in_folder}/train_set", train_set)
        np.save(f"{self.in_folder}/calib_set", calib_set)
        np.save(f"{self.in_folder}/test_set", test_set)

    def save_preprocessed(self) -> None:
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)
        torch.save([self.data, self.targets],
                   f"{self.out_folder}/{self.split}_processed.pt")

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train_set = CaliforniaHousing(split='train', in_folder=input_filepath,
                                  out_folder=output_filepath)
    train_set.save_preprocessed()
    calib_set = CaliforniaHousing(split='calib', in_folder=input_filepath,
                                  out_folder=output_filepath)
    calib_set.save_preprocessed()
    test_set = CaliforniaHousing(split='test', in_folder=input_filepath,
                                 out_folder=output_filepath)
    test_set.save_preprocessed()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
