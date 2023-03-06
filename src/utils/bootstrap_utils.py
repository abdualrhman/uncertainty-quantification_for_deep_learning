import time
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from functorch import combine_state_for_ensemble, vmap
from src.data.make_cifar10_dataset import CIFAR10, get_img_transformer
import torch.nn.functional as nnf
from src.utils.utils import get_metrics_score, accuracy


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


def train_accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def train_classifier(model, dataloader, print_acc: bool = False, num_epoches=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_accuracies = []
    step = 0
    for epoch in range(num_epoches):
        loss_tracker = []
        train_accuracies_batches = []
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            output = model(inputs.to(device))
            loss = criterion(output, targets.to(device))
            loss.backward()
            optimizer.step()
            step += 1
            loss_tracker.append(loss.item())
            predictions = output.max(1)[1]
            if print_acc:
                train_accuracies_batches.append(
                    train_accuracy(targets, predictions))
                if step % 500 == 0:
                    train_accuracies.append(np.mean(train_accuracies_batches))
                    print(
                        f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")


def train_ensemble(models, dataset, print_acc: bool = False, num_epoches: int = 10):
    for model in models:
        # get a random sample from dataset with size N
        random_train_idx = np.random.choice(
            np.array(range(len(dataset))), replace=False, size=25600)
        train_subset = torch.utils.data.Subset(dataset, random_train_idx)
        dataloader = torch.utils.data.DataLoader(train_subset, batch_size=32)
        train_classifier(model, dataloader, print_acc=True,
                         num_epoches=num_epoches)


def get_ensemble_preparation(models, minibatch):
    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    predictions2_vmap = vmap(fmodel, in_dims=(
        0, 0, None),  randomness='same')(params, buffers, minibatch)
    return predictions2_vmap


def get_softvotes(probs):
    sum_probs = torch.tensor([])
    for pred_idx in range(probs.shape[1]):
        pred_lis = torch.tensor([])
        for model_prob in probs:
            pred_lis = torch.cat(
                (pred_lis, model_prob[pred_idx].unsqueeze(dim=0)))
        sum_probs = torch.cat(
            (sum_probs, torch.sum(pred_lis, dim=0).unsqueeze(dim=0)))
    return sum_probs


def validate_ensemble(models: list, print_bool: bool, val_loader):
    with torch.no_grad():
        top1 = AverageMeter("top1")
        f1score = AverageMeter('f1score')
        batch_time = AverageMeter('batch_time')
        [model.eval() for model in models]
        end = time.time()
        N = 0
        for inputs, targets in val_loader:
            # compute ensemble output
            output = get_ensemble_preparation(models, inputs)

            probs = torch.stack([nnf.softmax(x, dim=1) for x in output])
            # validate shape and correctness
            assert output.shape == probs.shape
            assert torch.all(
                torch.eq(probs[0], nnf.softmax(output[0], dim=1))).item()
            output = get_softvotes(probs)
            top1_acc = accuracy(get_softvotes(probs), targets)[
                0].item()/100.0

            batch_f1_score, _, _ = get_metrics_score(
                output, targets)
            top1.update(top1_acc, n=inputs.shape[0])
            f1score.update(batch_f1_score, inputs.shape[0])
            batch_time.update(time.time() - end)
            N = N + inputs.shape[0]
            
            if print_bool:
                print(
                    f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} | F1: {f1score.val:.3f} |', end='')
    if print_bool:
        print('')
    return top1.avg, f1score.avg
