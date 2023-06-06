# Uncertainty Quantification in Deep Learning

This repository contains the code used in the production of results for the thesis "Uncertainty Quantification in Machine Learning Models Using Conformal Prediction".

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and saved models,
    │
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   └── tables         <- Generated LaTeX tables.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make.py
    │   │   ├── make_amzn_stock_price_dataset.py
    │   │   ├── make_housing_dataset.py
    │   │   ├── make_wine_quality_dataset.py
    │   │    make_cifar10_dataset.py
    │   │
    │   │
    │   ├── models  <- Models used for experiments
    │   │     ├── train_models  <- Scripts to train models and then use trained models
    |   |     ├── catBoost_quantile_regressor.py
    |   |     ├── cifar10_conv_model.py
    |   |     ├── conformal_classifier.py
    |   |     ├── conformal_forcast.py
    |   |     ├── CQR.py
    |   |     ├── gradient_boosting_quantile_regressor.py
    |   |     ├── lstm_model.py
    |   |     ├── oracle.py  <- Base class used for active learning
    |   |     └── quantile_net.py
    |   |
    │   │
    │   └── visualization  <- Scripts to generate LaTeX tables and figures from experiment results
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Usage

### Requirments

To install requirements:

```
pip install -r requirements.txt
```

## Download pretrained models

In order run the experiments, you must download the pretrained models by using:

```
make models
```

### Reproduce the tables and figures

To reproduce the tables, type the following command

```
make table-<TABLE_NUMBER>
```

For example:

```
make table-2.1
```

or

```
table-5.1.a
```

To reproduce figures, use the following command

```
make figure-<FIGURE_NUMBER>
```

### Using the conformal prediction methods

The class `ConformalClassifier` is built to conformalize a neural network clas-
sifier, taking a model, calibration dataloader and an error rate α as parameters
and outputs prediction sets. Upon initialization, the class calibrates the model
using the score function defined in equation 2.5 and computes ˆq. During the
forward pass, the class constructs a prediction set from the base model outputs.

```python
import torch
import torchvision
from src.models.conformal_classifier import ConformalClassifier
from src.utils.utils import get_CIFAR10_img_transformer
# load datasets
calib_set = torchvision.datasets.CIFAR10(train=True, root='data/processed', download=True, transform=get_CIFAR10_img_transformer())
test_set = torchvision.datasets.CIFAR10(train=False, root='data/processed', download=True, transform=get_CIFAR10_img_transformer())

calib_loader = torch.utils.data.DataLoader(calib_set)
test_loader = torch.utils.data.DataLoader(test_set)
# conformalize
pretrained_model = Cifar10ConvModel()
conformal_model = ConformalClassifier(model=pretrained_model, calib_loader=calib_loader, alpha=0.1)
# prediction
for x, y in test_loader:
    prediction_set = conformal_model(x)
```

Similar to above, the class `CQR` is built to conformalize a quantile neural net-
work regressor, taking a model, calibration dataloader and an error rate α as
parameters and outputs prediction intervals. Upon initialization, the class cali-
brates the model using the score function defined in equation 3.7 and computes
ˆq. During the forward pass, the class constructs a prediction interval from the
base model outputs.

```python
import torch
from src.models.CQR import CQR
from src.models.quantile_net import QuantileNet
from src.data.make_wine_quality_dataset import WineQuality
# load datasets
calib_set = WineQuality(train=True, in_folder='data/raw', out_folder='data/processed')
test_set = WineQuality(train=False, in_folder='data/raw', out_folder='data/processed')

calib_loader = torch.utils.data.DataLoader(calib_set)
test_loader = torch.utils.data.DataLoader(test_set)
# conformalize
pretrained_model = QuantileNet(input_size=11)
conformal_model = CQR(model=pretrained_model,calib_loader=calib_loader, alpha=0.1)
# prediction
for x, y in test_loader:
    prediction_set = conformal_model(x)
```

The `Pinball loss` function in equation 3.5 is costume build and can be used by
passing a list of desired quantiles as a parameter.

```python
from src.models.quantile_net import QuantileLoss

loss_fn = QuantileLoss([0.05, 0.95])
```

For active learning, the class `Oracle` can be used by passing the model, training set, test set, and sampling parameters.

```python

import torchvision
from src.models.oracle import Oracle
from src.models.cifar10_conv_model import Cifar10ConvModel
from src.utils.utils import get_CIFAR10_img_transformer
# load datasets
train_set = torchvision.datasets.CIFAR10(
    train=True, root='data/processed', transform=get_CIFAR10_img_transformer())
test_set = torchvision.datasets.CIFAR10(
    train=False, root='data/processed', transform=get_CIFAR10_img_transformer())

model = Cifar10ConvModel()
oracle = Oracle(model=model, train_set=train_set, test_set=test_set, strategy='least-confidence', sample_size=1000)
# start active learning for 1 round
oracle.teach(1)
print(oracle.round_accuracies)
```

## Contact

If you have any questions or feedback, please feel free to reach out at `abdulrahman.ramdan@outlook.com`.
