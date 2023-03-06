
from ensurepip import bootstrap
from numpy import outer
from src.utils.bootstrap_utils import *
from src.utils.utils import *
from src.models.cifar10_conv_model import Cifar10ConvModel
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import pandas as pd
import random

from src.utils.cp_utils import get_model

# models = [Cifar10ConvModel(), Cifar10ConvModel()]
# train_ensemble(models, print_acc=True, num_epoches=3)
# data = torch.randn(1, 3, 32, 32)

# pred = get_ensemble_preparation(models, data)

# print(pred.shape)
# print(pred[1].max(1)[1])
# print("pred[1]")
# print(pred[1])


def experiment(models: list, datasetname: str, num_trials: int, bsz: int, num_train_epoch: int, train_bool):
    top1s = np.zeros((num_trials,))
    f1s = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        top1_avg, f1score_avg = trial(
            models,  datasetname, bsz, num_train_epoch, train_bool)
        top1s[i] = top1_avg
        f1s[i] = f1score_avg
        print(
            f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, F1: {np.median(f1s[0:i+1]):.3f} \033[F', end='')
    print('')
    return np.median(top1s), np.median(f1s)


def trial(modelnames: list, datasetname: str, bsz: int, num_train_epoch: int, train_bool: bool = True):
    models = [get_model(name) for name in modelnames]
    if train_bool:
        train_set = get_dataset(datasetname, split='train')
        train_ensemble(models, train_set, print_acc=True,
                       num_epoches=num_train_epoch)
    test_set = get_dataset(datasetname, split='test')
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bsz, shuffle=True)
    top1_avg, f1score_avg = validate_ensemble(
        models, val_loader=val_loader, print_bool=False)
    return top1_avg, f1score_avg


if __name__ == "__main__":
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cache_fname = "./.cache/bootstrap_cifar10_df.csv"
    try:
        df = pd.read_csv(cache_fname)
    except:
        modelnames = ['Cifar10ConvModel']
        datasetname = 'Cifar10'
        num_trials = 2
        bsz = 32
        bootstrap_splits = 1
        num_train_epoch = 1
        train_bool = False
        cudnn.benchmark = True
        df = pd.DataFrame(columns=["Model", "Top1", "F1"])
        for modelname in modelnames:
            print(f'Model: {modelname}')
            out = experiment(modelnames, datasetname,
                             num_trials, bsz, num_train_epoch, train_bool)
            print(out)
            df = df.append(
                {"Model": modelname, "Top1": np.round(out[0], 3), "F1": np.round(out[1], 3)}, ignore_index=True)
        df.to_csv(cache_fname)
