
import torch
import time
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple
from src.utils.utils import isTorchModel, AverageMeter
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def validate_forcaster(val_loader,  model, print_bool: bool = False) -> Tuple[float, float]:
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        length = AverageMeter('length')
        coverage = AverageMeter('coverage')
        end = time.time()
        N = 0
        for (x, target) in val_loader:
            target = target.numpy()
            S = model(x)
            low_bound = S[0]
            up_bound = S[1]

            leng = np.abs(up_bound - low_bound)
            cov = regression_coverage_score(
                y_true=target, y_pred_low=low_bound, y_pred_up=up_bound)

            length.update(leng.mean(), n=x.shape[0])
            coverage.update(cov, n=x.shape[0])
            batch_time.update(time.time() - end)
            N = N + x.shape[0]

            if print_bool:
                print(
                    f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Length: {length.val:.3f} ({length.avg:.3f}) | Cov: {coverage.val:.3f} ({coverage.avg:.3f})', end='')
    if print_bool:
        print('')  # Endline
    return length.avg, coverage.avg


def regression_coverage_score(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike,
) -> float:
    coverage = np.mean(
        ((y_pred_low <= y_true) & (y_pred_up >= y_true))
    )
    return float(coverage)
