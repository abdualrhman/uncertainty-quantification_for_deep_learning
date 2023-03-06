import torch
from torch import nn
from src.data.make_cifar10_dataset import CIFAR10, get_img_transformer
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.cifar10_conv_model import Cifar10ConvModel
from sklearn import metrics
import numpy as np
import torch.nn.functional as nnf
from skimage import io
from src.models.conformal_model import ConformalModel, ConformalModelLogits
from src.utils.cp_utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
test_dataset = CIFAR10(split="test", root='data/processed', download=False,
                       transform=get_img_transformer())
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32)


premodel = Cifar10ConvModel().cpu()
premodel.load_state_dict(torch.load("models/trained_model.pt"))
calib_dataset = CIFAR10(split="calib", root='data/processed', download=False,
                        transform=get_img_transformer())
calib_loader = torch.utils.data.DataLoader(
    calib_dataset, batch_size=32)
conf_model = ConformalModel(
    model=premodel, calib_loader=calib_loader, alpha=0.05)

validate(test_dataloader, conf_model, True)
