
from torchvision import transforms
import torchvision
from src.models.cifar10_conv_model import Cifar10ConvModel
from src.models.oracle import Oracle
from tests import _PATH_DATA, _PATH_SRC
import numpy as np
import pytest
import sys
sys.path.append('../src')


train_set = torchvision.datasets.CIFAR10(train=True, root='../data/processed', transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), download=True)
test_set = torchvision.datasets.CIFAR10(train=False,  root='../data/processed',  transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), download=True)


def test_number_labeled_data():
    params = {
        'sample_size': 150,
        'strategy': 'least-confidence'
    }
    model = Cifar10ConvModel()
    o = Oracle(model=model, train_set=train_set, test_set=test_set, **params)
    # all labeled index are set to Flase
    assert all(o.labeled_idx_bool) == False
    o.teach(1)
    # test number of labeled data after one learning round
    assert len(
        o.labeled_idx_bool[o.labeled_idx_bool == True]) == params['sample_size']
    assert len(o.get_labeled_data()[0] ==
               params['sample_size']) == params['sample_size']
    # test number of unlabeled data after one learning round
    assert len(o.get_unlabeled_data()[0] ==
               params['sample_size']) == len(train_set) - params['sample_size']

    o.teach(1)
    # test number of labeled data after two learning round
    assert len(
        o.labeled_idx_bool[o.labeled_idx_bool == True]) == params['sample_size'] * 2
    assert len(o.get_labeled_data()[0] ==
               params['sample_size']) == params['sample_size'] * 2

    # test number of unlabeled data after two learning round
    assert len(o.get_unlabeled_data()[0] ==
               params['sample_size']) == len(train_set) - (params['sample_size'] * 2)


def test_update_labeled_data_method():
    params = {
        'sample_size': 150,
        'strategy': 'least-confidence'
    }
    model = Cifar10ConvModel()
    o = Oracle(model=model, train_set=train_set, test_set=test_set, **params)
    assert all(o.labeled_idx_bool) == False
    queried_idx = np.sort(np.random.randint(
        len(train_set), size=(params['sample_size'])))
    o.update_labeled_data(queried_idx)
    # check size
    assert len(o.get_labeled_data()[0]) == params['sample_size']
    # check the exact indecies are updated
    assert all(queried_idx == o.get_labeled_data()[0])


def test_teaching_with_too_many_rounds_raises_error():
    params = {
        'sample_size': 150,
        'strategy': 'least-confidence'
    }
    model = Cifar10ConvModel()
    o = Oracle(model=model, train_set=train_set, test_set=test_set, **params)
    with pytest.raises(ValueError):
        o.teach(100000)


def test_query_sample_idx_are_unlabeled():
    params = {
        'sample_size': 150,
        'strategy': 'least-confidence'
    }
    model = Cifar10ConvModel()
    o = Oracle(model=model, train_set=train_set, test_set=test_set, **params)
    o.teach(1)
    sample_idx = o.query_sample_idx()
    labeled_idx = o.get_unlabeled_data()[0]
    # check queried indecies are unlabeled
    assert np.in1d(labeled_idx, sample_idx()).shape[0] == 0
