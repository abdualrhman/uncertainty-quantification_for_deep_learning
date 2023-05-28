import warnings
import torch
import time
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple
from src.models.CQR import CQR_logits
from src.utils.utils import get_model_output, isTorchModel, AverageMeter
from src.models.gradient_boosting_quantile_regressor import ConformalPreComputedLogits


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", UserWarning)


def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(
        dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(
        temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2


def conformalize_regressor(model, loader_cal, alpha: float):
    if isTorchModel(model):
        return CQR_logits(
            model, loader_cal, alpha=alpha)
    else:
        return ConformalPreComputedLogits(alpha=alpha).fit_precomputed_logits(loader_cal)


def validate(val_loader, cal_loader,  model, alpha: bool, precomputed_logits: bool = False, print_bool: bool = False) -> Tuple[float, float]:
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        length = AverageMeter('length')
        coverage = AverageMeter('coverage')
        if not precomputed_logits and isTorchModel(model):
            model.eval()
        end = time.time()
        N = 0
        for (x, target) in val_loader:
            target = target.detach().cpu().numpy()
            if precomputed_logits:
                S = x.T.detach().cpu().numpy()
            else:
                model = conformalize_regressor(model, cal_loader, alpha)
                S = get_model_output(model, x.cpu()).detach().cpu().numpy()
            q_lo, q_hi = S
            leng = np.abs(q_hi - q_lo)
            cov = regression_coverage_score(target, q_lo, q_hi)

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
