import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import os
import warnings
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import torchvision
from torchvision import transforms
from src.data.make_housing_dataset import CaliforniaHousing
from src.data.make_wine_quality_dataset import WineQuality
from src.models.gradient_boosting_quantile_regressor import GBQuantileRegressor
from src.models.catBoost_quantile_regressor import CatBoostQunatileRegressor
from src.models.cifar10_conv_model import Cifar10ConvModel
from src.models.lstm_model import LSTM
from src.data.make_amzn_stock_price_dataset import AMZN_SP
from src.models.quantile_net import QuantileNet
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def get_model_alpha(modelname: str) -> str:
    return float(modelname.split(":")[-1])


def get_aug_img_transformer():
    return transforms.Compose([
        transforms.AugMix(severity=6, chain_depth=6),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_CIFAR10_img_transformer():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]
    )


params = {
    'CIFAR10': {
        'train_args': {'batch_size': 64, 'num_workers': 0},
        'test_args': {'batch_size': 1000, 'num_workers': 0},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5},
    }
}


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, datapoints):
        super().__init__()
        self.data = datapoints[0]
        self.targets = datapoints[1]

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_dataset(datasetname: str, train: bool = True, datapath: str = 'data/processed'):
    split = 'train' if train else 'test'
    if datasetname == 'Cifar10':
        return torchvision.datasets.CIFAR10(train=train, root=datapath, download=True,
                                            transform=get_CIFAR10_img_transformer())
    elif datasetname == 'Cifar10Aug':
        return torchvision.datasets.CIFAR10(train=train, root=datapath, download=True,
                                            transform=get_aug_img_transformer())
    elif datasetname == 'california_housing':
        return CaliforniaHousing(
            split=split, in_folder='data/raw', out_folder='data/processed')
    elif datasetname == 'wine_quality':
        return WineQuality(
            train, in_folder='data/raw', out_folder='data/processed')
    elif datasetname == 'AMZN':
        return AMZN_SP(
            train, in_folder='data/raw', out_folder='data/processed')


def get_metrics_score(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    f1 = f1_score(target.detach().cpu(),
                  pred.detach().cpu()[0], average='macro', zero_division=0)
    precision = precision_score(target.detach().cpu(),
                                pred.detach().cpu()[0], average='macro', zero_division=0)
    recall = recall_score(target.detach().cpu(),
                          pred.detach().cpu()[0], average='macro', zero_division=0)
    return f1, precision, recall


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_logits_targets(model, loader, out_dim=1):
    """
    Compute model's logits on a given dataloader

    Parameters
    ----------
    model: Model
    loader: Dataloader
    out_dim: int

    Returns
    -------
    dataset_logits: Dataset
    """
    logits = torch.zeros((len(loader.dataset), out_dim))
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = get_model_output(model, x.cpu())
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())

    return dataset_logits


def get_model_output(model, inputs):
    """
    Get uniform tensor output for implemented models
    """
    if isTorchModel(model):
        return torch.FloatTensor(model(inputs))
    else:
        return torch.FloatTensor(model.predict(inputs))


def isTorchModel(model) -> bool:
    return hasattr(model, 'parameters')


def get_out_dataset_dim(datasetname: str) -> int:
    """
    Get the model's output dimension for a given dataset

    Parameters
    ----------
    datasetname : str

    Returns
    -------
    out_dim: int 
    """
    if datasetname == 'Cifar10' or 'Cifar10Aug':
        return 10
    elif datasetname == 'california_housing':
        return 2
    elif datasetname == 'wine_quality':
        return 2
    else:
        raise Exception("Unknown dataset")


def get_logits_dataset(modelname, datasetname, cache='.cache/'):
    fname = cache + datasetname + '_precomputed_logits' + '/' + modelname + '.pkl'
    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)
    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(modelname, datasetname)
    dataset = get_dataset(datasetname, train=False)
    # Check model type to get models logits and targets
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32)
    dataset_logits = get_logits_targets(
        model, loader,  get_out_dataset_dim(datasetname))
    # Save the dataset
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits


def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name, getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)


def get_untrained_model(modelname: str):
    if modelname == 'Cifar10ConvModel':
        model = Cifar10ConvModel()
    return model


def get_model(modelname, datasetname=''):
    if modelname == 'Cifar10Resnet20':
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20",  pretrained=True)
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    elif modelname == "Cifar10ConvModel":
        model = Cifar10ConvModel()
        model.load_state_dict(torch.load(
            "./models/trained_conv_cifar10_model.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    elif modelname == "QuantileNet:0.05":
        input_size = 8 if datasetname == 'california_housing' else 11
        model = QuantileNet(input_size=input_size)
        model.load_state_dict(torch.load(
            f"./models/trained_quantile_net0.05_{datasetname}.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()
    elif modelname == "QuantileNet:0.1":
        input_size = 8 if datasetname == 'california_housing' else 11
        model = QuantileNet(input_size=input_size)
        model.load_state_dict(torch.load(
            f"./models/trained_quantile_net0.1_{datasetname}.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()
    elif modelname == "QuantileNet:0.2":
        input_size = 8 if datasetname == 'california_housing' else 11
        model = QuantileNet(input_size=input_size)
        model.load_state_dict(torch.load(
            f"./models/trained_quantile_net0.2_{datasetname}.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    elif modelname == "GradientBoostingRegressor:0.05":
        with open(f'./models/trained_gradientboost_regressor0.05_{datasetname}.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open(f'./models/trained_gradientboost_regressor0.95_{datasetname}.pkl', 'rb') as p:
            model_up = pickle.load(p)
        model = GBQuantileRegressor(
            low=model_low, up=model_up)
    elif modelname == "GradientBoostingRegressor:0.1":
        with open(f'./models/trained_gradientboost_regressor0.1_{datasetname}.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open(f'./models/trained_gradientboost_regressor0.9_{datasetname}.pkl', 'rb') as p:
            model_up = pickle.load(p)
        model = GBQuantileRegressor(
            low=model_low, up=model_up)
    elif modelname == "GradientBoostingRegressor:0.2":
        with open(f'./models/trained_gradientboost_regressor0.2_{datasetname}.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open(f'./models/trained_gradientboost_regressor0.8_{datasetname}.pkl', 'rb') as p:
            model_up = pickle.load(p)
        model = GBQuantileRegressor(
            low=model_low, up=model_up)

    elif modelname == "CatBoostingRegressor:0.05":
        with open(f'./models/trained_catboost_regressor0.05_{datasetname}.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open(f'./models/trained_catboost_regressor0.95_{datasetname}.pkl', 'rb') as p:
            model_up = pickle.load(p)
        model = CatBoostQunatileRegressor(
            low=model_low, up=model_up)
    elif modelname == "CatBoostingRegressor:0.1":
        with open(f'./models/trained_catboost_regressor0.1_{datasetname}.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open(f'./models/trained_catboost_regressor0.9_{datasetname}.pkl', 'rb') as p:
            model_up = pickle.load(p)
        model = CatBoostQunatileRegressor(
            low=model_low, up=model_up)
    elif modelname == "CatBoostingRegressor:0.2":
        with open(f'./models/trained_catboost_regressor0.2_{datasetname}.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open(f'./models/trained_catboost_regressor0.8_{datasetname}.pkl', 'rb') as p:
            model_up = pickle.load(p)
        model = CatBoostQunatileRegressor(
            low=model_low, up=model_up)

    elif modelname == 'LSTM_AMZN':
        model = LSTM(1, 4, 2)
        model.load_state_dict(torch.load("./models/trained_lstm_amzn.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    else:
        print(modelname)
        raise NotImplementedError

    return model


def pre_transform_dataset(dataset: Dataset) -> Dataset:
    """
    Pre-transform a givin dataset

    Parameters
    ----------
    dataset: Dataset

    Returns
    -------
    pre_transformed_dataset: Dataset
    """
    train_loader = DataLoader(
        dataset, batch_size=len(dataset))
    datapoints = []
    for i in train_loader:
        datapoints = i

    return TransformedDataset(datapoints)


def get_lstm_model_std(inputs: torch.Tensor, num_repeats: int = 10, dropout_fraction: float = 0.4):
    """
    Compute model output variance using random dropout

    Parameters
    ----------
    inputs: Tensor
    num_repeats: int
    dropout_fraction: float

    Returns
    -------
    variance: float
    """
    # overwrite model droupout
    model = LSTM(1, 4, 2, dropout_fraction)
    model.load_state_dict(torch.load("models/trained_lstm_amzn.pt"))
    model = torch.nn.DataParallel(model).cpu()

    outputs = []
    for _ in range(num_repeats):
        output = model(inputs)
        outputs.append(output)

    return torch.std(torch.stack(outputs), dim=0)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_monte_carlo_prediction_sets(val_loader,
                                    forward_passes,
                                    model,
                                    n_classes,
                                    n_samples, alpha, randomized=True):
    """Get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    alpha: float

    Returns
    ---------
    mean : ArrayLike
        class probability across predictions
    S : ArrayLike
        (1-alpha)% prediction set

    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(val_loader):
            image = image.to(torch.device('cpu'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
    # make prediction sets
    S = []
    for pred_prob in mean:
        sorted_classes = np.argsort(-pred_prob)
        prediction_set = []
        total_probability = 0.0

        for class_ in sorted_classes:
            prediction_set.append(class_)
            total_probability += pred_prob[class_]
            if total_probability >= 1-alpha:
                # randomly remove last class from prediction set
                if randomized and np.random.randint(2) == 0 and len(prediction_set) > 1:
                    prediction_set.pop()
                break
        S.append(prediction_set)
    return mean, S


def validate_monte_carlo_prediction(val_loader, predictions_sets):
    """ Validate monte-carlor prediction sets

    Parameters
    ----------
    val_loader : object
        data loader object from the data loader module
    predictions_sets : ArrayLike
        monte-carlo prediction sets

    Returns
    ---------
    coverage : float
    set_size : float
    f1_score : float
    """
    size = AverageMeter('size')
    coverage = AverageMeter('coverage')
    f1_scores = AverageMeter('f1_scores')
    pred_set_loader = torch.utils.data.DataLoader(
        predictions_sets, batch_size=val_loader.batch_size, shuffle=False, collate_fn=lambda x: x)
    for S, (_, y) in zip(pred_set_loader, val_loader):
        top1s = []
        [top1s.append(p_set[0]) for p_set in S]
        S_tensor = [torch.tensor(sublist) for sublist in S]
        cov, sz = coverage_size(S_tensor, y)
        f1 = f1_score(y_true=y.detach().cpu(),
                      y_pred=top1s, average='macro', zero_division=0)
        coverage.update(cov)
        size.update(sz)
        f1_scores.update(f1)
    return coverage.avg, size.avg, f1_scores.avg


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]
