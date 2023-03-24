

from ensurepip import bootstrap
from numpy import outer
from src.utils.bootstrap_utils import *
from src.utils.utils import *
import torch.backends.cudnn as cudnn
import pandas as pd
import random
from src.experiments.bootstrap import experiment

if __name__ == "__main__":
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cache_fname = "./.cache/bootstrap_aug_cifar10_df.csv"
    try:
        df = pd.read_csv(cache_fname)
    except:
        modelnames = ['Cifar10ConvModel']
        datasetname = 'Cifar10Aug'
        num_trials = 10
        bsz = 32
        bootstrap_splits = 1
        num_train_epoch = 1
        train_bool = False
        cudnn.benchmark = True

        train_set = None if train_bool else get_dataset(
            datasetname, split='train')
        test_set = get_dataset(datasetname, split='test')
        df = pd.DataFrame(columns=["Model", "Top1", "F1"])
        for modelname in modelnames:
            print(f'Model: {modelname}')
            out = experiment(modelnames, train_set, test_set,
                             num_trials, bsz, num_train_epoch, train_bool)
            print(out)
            df = df.append(
                {"Model": modelname, "Top1": np.round(out[0], 3), "F1": np.round(out[1], 3)}, ignore_index=True)
        df.to_csv(cache_fname)
