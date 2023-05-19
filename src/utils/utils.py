import os
import torch
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import torchvision
from torchvision import transforms
# from src.data.make_cifar10_dataset import CIFAR10, get_aug_img_transformer
from src.data.make_housing_dataset import CaliforniaHousing
from src.models.GB_quantile_regressor import GB_quantile_regressor
from src.models.cifar10_conv_model import Cifar10ConvModel
from src.models.lstm_model import LSTM
from src.data.make_amzn_stock_price_dataset import AMZN_SP
from src.models.regFNN import RegFNN
from torch.utils.data import Dataset, DataLoader


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
    if datasetname == 'Cifar10':
        return torchvision.datasets.CIFAR10(train=train, root=datapath, download=True,
                                            transform=get_CIFAR10_img_transformer())
    elif datasetname == 'Cifar10Aug':
        return torchvision.datasets.CIFAR10(train=train, root=datapath, download=True,
                                            transform=get_aug_img_transformer())
    elif datasetname == 'CalHousing':
        return CaliforniaHousing(
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
    Compute model's logits 

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
            batch_logits = torch.tensor(
                get_model_output(model, x.cpu()))
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())

    return dataset_logits


def get_model_output(model, inputs):
    """
    Get uniform output for implemented models
    """
    if isTorchModel(model):
        return model(inputs)
    else:
        return model.predict(inputs)


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
    if datasetname == 'Cifar10':
        return 10
    elif datasetname == 'CalHousing':
        return 2
    else:
        raise Exception("Unknown dataset")


def get_logits_dataset(modelname, datasetname, datasetpath='', cache='src/experiments/.cache/'):
    fname = cache + datasetname + '/' + modelname + '.pkl'
    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)
    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(modelname)
    dataset = get_dataset(datasetname, split='test')
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


def get_model(modelname):
    if modelname == 'Cifar10Resnet20':
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20",  pretrained=True)
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    elif modelname == "Cifar10ConvModel":
        model = Cifar10ConvModel()
        model.load_state_dict(torch.load("./models/trained_model.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    elif modelname == "RegFFN":
        model = RegFNN()
        model.load_state_dict(torch.load("./models/trained_reg_fnn.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    elif modelname == "GBQuantileReg":
        with open('./models/trained_gbreg0.05.pkl', 'rb') as p:
            model_low = pickle.load(p)
        with open('./models/trained_gbreg0.95.pkl', 'rb') as p:
            model_up = pickle.load(p)
        with open('./models/trained_gbreg0.5.pkl', 'rb') as p:
            model_median = pickle.load(p)
        model = GB_quantile_regressor(
            low=model_low, up=model_up, median=model_median)

    elif modelname == 'LSTM_AMZN':
        model = LSTM(1, 4, 2)
        model.load_state_dict(torch.load("./models/trained_lstm_amzn.pt"))
        model.eval()
        model = torch.nn.DataParallel(model).cpu()

    else:
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
