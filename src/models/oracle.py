

import numpy as np
import torch
from typing import Sequence
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch.optim as optim

from src.models.conformal_model import ConformalModel


class DataSubset(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence) -> None:
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], idx
        return self.dataset[self.indices[idx]], idx


class Oracle:
    def __init__(self, model, train_set: Dataset, test_set: Dataset, sample_size: int, strategy: str, n_init_training_labels: int = 0, n_training_epochs: int = 2, alpha: float = 0.1, calib_size: int = 500, train_on_all_unlabeled: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_set = train_set
        self.test_set = test_set
        self.strategy = strategy
        self.sample_size = sample_size
        self.n_init_training_labels = n_init_training_labels
        self.n_training_epochs = n_training_epochs
        self.alpha = alpha
        self.calib_size = calib_size
        self.train_on_all_unlabeled = train_on_all_unlabeled
        self.labeled_idx_bool = np.zeros(len(train_set), dtype=bool)
        self.round_accuracies = []

        self.model.to
        if self.strategy.startswith("conformal-score"):
            if calib_size >= len(train_set):
                raise ValueError(
                    "Invalid calibration size")
            # conformalize the model
            self.init_conformal_model()

        if n_init_training_labels > 0:
            self.init_trainig()

    def get_labeled_data(self):
        unlabeled_idx = np.arange(len(self.labeled_idx_bool))[
            self.labeled_idx_bool]

        return unlabeled_idx, DataSubset(self.train_set, unlabeled_idx)

    def get_unlabeled_data(self):
        labeled_idx = np.arange(len(self.labeled_idx_bool))[
            ~self.labeled_idx_bool]
        return labeled_idx,  DataSubset(self.train_set, labeled_idx)

    def initialize_training_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(len(self.train_set))
        np.random.shuffle(tmp_idxs)
        self.labeled_idx_bool[tmp_idxs[:num]] = True

    def accuracy(self, target, pred):
        return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

    def train_model(self, train_set):
        # _, train_data = self.get_labeled_data()
        train_loader = torch.utils.data.DataLoader(
            train_set, shuffle=False, batch_size=32)
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(self.n_training_epochs):
            for (x, y), _ in train_loader:
                optimizer.zero_grad()
                output = self.model(x.to(self.device))
                loss = criterion(output, y.to(self.device))
                loss.backward()
                optimizer.step()

    def get_model_accuracy(self):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=len(self.test_set))
        acc = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                out = self.model(x)
                preds = out.max(1)[1]
                acc = self.accuracy(y, preds)
            return acc

    def get_model_predictions(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False)
        preds = torch.zeros(
            [len(dataset), len(np.unique(self.train_set.targets))])
        self.model.eval()
        with torch.no_grad():
            for (x, _), _ in dataloader:
                out = self.model(x)
                prob = F.softmax(out, dim=1)
                preds = prob.cpu()
        # self.model.train()
        return preds

    def get_model_conformal_predictions(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False)
        preds = torch.zeros(
            [len(dataset), len(np.unique(self.train_set.targets))])
        self.conformal_model.eval()
        with torch.no_grad():
            for (x, _), idx in dataloader:
                logits, _ = self.conformal_model(x)
                # prob = F.softmax(logits, dim=1)
                preds[idx] = logits.cpu()
        self.model.train()
        return preds

    def check_valid_available_data_size(self, n_rounds: int) -> None:
        unlabeled_idx, _ = self.get_unlabeled_data()
        if len(unlabeled_idx) < n_rounds * self.sample_size:
            raise ValueError(
                "Too few unlabeled data remained. Reduce number of learning rounds or sample size")

    def update_labeled_data(self, labeled_idx) -> None:
        self.labeled_idx_bool[labeled_idx] = True

    def init_trainig(self) -> None:
        print(
            f"Initial model training with {self.n_init_training_labels} samples")
        self.initialize_training_labels(num=self.n_init_training_labels)
        trian_idx, _ = self.get_labeled_data()
        train_set = self.get_train_data(trian_idx)
        self.train_model(train_set)

    def teach(self, n_rounds: int) -> None:
        self.check_valid_available_data_size(n_rounds)
        for i in tqdm(range(n_rounds)):
            self.round_accuracies.append(self.get_model_accuracy())
            sample_idx = self.query_sample_idx()
            self.update_labeled_data(sample_idx)
            train_set = self.get_train_data(sample_idx)
            self.train_model(train_set)

    def init_conformal_model(self):
        calib_idx = np.random.randint(low=0, high=len(
            self.train_set), size=self.calib_size)
        # remove from training pool
        self.update_labeled_data(calib_idx)
        calib_subset = torch.utils.data.Subset(self.train_set, calib_idx)
        calib_loader = torch.utils.data.DataLoader(
            calib_subset, shuffle=False)
        self.conformal_model = ConformalModel(
            model=self.model, calib_loader=calib_loader, alpha=self.alpha)

    def get_train_data(self, train_idx=None):
        if self.train_on_all_unlabeled:
            _, train_set = self.get_labeled_data()
            return train_set
        else:
            return DataSubset(self.train_set, train_idx)

    def query_sample_idx(self):
        if self.strategy == 'least-confidence':
            idx, train_data = self.get_unlabeled_data()
            preds = self.get_model_predictions(train_data)
            uncertainties = preds.max(1)[0]
            return idx[uncertainties.sort()[1][:self.sample_size]]

        elif self.strategy == 'random-sampler':
            return np.random.choice(np.where(self.labeled_idx_bool == False)[0], self.sample_size, replace=False)

        elif self.strategy == 'conformal-score:least-confidence':
            idx, data = self.get_unlabeled_data()
            conformal_predictions = self.get_model_conformal_predictions(data)
            uncertainties = conformal_predictions.max(1)[0]
            return idx[uncertainties.sort()[1][:self.sample_size]]

        elif self.strategy == 'entropy-sampler':
            idx, train_data = self.get_unlabeled_data()
            preds = self.get_model_predictions(train_data)
            log_probs = torch.log(preds)
            entropy_scores = (preds*log_probs).sum(1)
            return idx[entropy_scores.sort()[1][:self.sample_size]]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
