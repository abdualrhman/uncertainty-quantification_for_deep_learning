
import pickle5 as pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import pandas as pd

from src.data.make_wine_quality_dataset import WineQuality
from src.data.make_housing_dataset import CaliforniaHousing
from src.models.catBoost_quantile_regressor import CatBoostQunatileRegressor
from src.utils.cp_reg_utils import regression_coverage_score


test_set = CaliforniaHousing(
    split='test', in_folder='data/raw', out_folder='data/processed')

datasetname = 'california_housing'
with open(f'./models/trained_catboost_reg0.05_{datasetname}.pkl', 'rb') as p:
    model_low = pickle.load(p)
with open(f'./models/trained_catboost_reg0.95_{datasetname}.pkl', 'rb') as p:
    model_up = pickle.load(p)
with open(f'./models/trained_catboost_reg0.5_{datasetname}.pkl', 'rb') as p:
    model_median = pickle.load(p)
model = CatBoostQunatileRegressor(
    low=model_low, up=model_up, median=model_median)


print(model.predict(test_set.data.numpy()))
