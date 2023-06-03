
import pickle5 as pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from src.utils.utils import get_CIFAR10_img_transformer
import pandas as pd
from src.models.cifar10_conv_model import Cifar10ConvModel
from src.models.oracle import Oracle

train_set = torchvision.datasets.CIFAR10(
    train=True, download=True, root='data/processed', transform=get_CIFAR10_img_transformer())
test_set = torchvision.datasets.CIFAR10(
    train=False, download=True, root='data/processed', transform=get_CIFAR10_img_transformer())
model = Cifar10ConvModel()
l = Oracle(model, train_set=train_set, sample_size=1000, test_set=test_set,
           n_init_training_labels=1000, strategy="conformal-prediction:least-confidence")


idx, data = l.get_unlabeled_data()
preds = l.get_model_conformal_predictions(data)
uncertainties = preds.max(1)[0]

