from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
import torch
import pandas as pd
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import lightning.pytorch as pl
from pathlib import Path
import copy
import os
import warnings
import pickle

# warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# load data

data = get_stallion_data()

# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# add additional features
data["month"] = data.date.dt.month.astype(str).astype(
    "category")  # categories have be strings
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = data.groupby(
    ["time_idx", "sku"], observed=True).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(
    ["time_idx", "agency"], observed=True).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = data[special_days].apply(
    lambda x: x.map({0: "-", 1: x.name})).astype("category")

# create dataset and dataloaders
max_prediction_length = 6
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    # group of categorical variables can be treated as one variable
    variable_groups={"special_days": special_days},
    time_varying_known_reals=["time_idx",
                              "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True)

batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0)

# evaluate baseline model
predictions = Baseline().predict(val_dataloader)
metric = MAE()
model = Baseline()
for x, y in val_dataloader:
    metric.update(model(x).prediction, y)
baseline_mae = metric.compute().item()

# train model
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")
pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="cpu",
    gradient_clip_val=0.0173,
    max_epochs=1,
    enable_model_summary=True,
    limit_train_batches=50,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    default_root_dir='./models/'
)

# init network
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.004,
    hidden_size=28,
    attention_head_size=3,
    dropout=0.13,
    hidden_continuous_size=11,
    loss=QuantileLoss(),
    optimizer="Ranger",
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


print("start training ...")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
print("finished training")

# model_save_path = "models/tft_forcaster.pkl"
# print(f"saving trained model {model_save_path}")
# with open(model_save_path, "wb") as fout:
#     pickle.dump(tft, fout)
