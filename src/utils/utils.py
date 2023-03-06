
from src.data.make_cifar10_dataset import CIFAR10, get_img_transformer, get_aug_img_transformer
import os
from sklearn.metrics import f1_score, precision_score, recall_score


def get_dataset(datasetname: str, split: str, datapath: str = 'data/processed'):
    if datasetname == 'Cifar10':
        return CIFAR10(split=split, root=datapath, download=True,
                       transform=get_img_transformer())
    elif datasetname == 'Cifar10Aug':
        return CIFAR10(split=split, root=datapath, download=True,
                       transform=get_aug_img_transformer())


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
