
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from src.utils.utils import get_logits_targets


class CQR(nn.Module):
    def __init__(self, model, calib_loader, alpha, batch_size=32):
        super(CQR, self).__init__()

        self.model = model
        # compute logits
        calib_logits_set = get_logits_targets(self.model, calib_loader)
        calib_logit_loader = torch.utils.data.DataLoader(
            calib_logits_set, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.alpha = alpha
        self.Qhat = conformal_calibration_logits(
            self, calib_logit_loader, len(calib_logits_set))

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()

            S = [logits_numpy[:, 0] - self.Qhat,
                 logits_numpy[:, 1] + self.Qhat]
        return S


class CQR_logits(nn.Module):
    def __init__(self, model, calib_loader, alpha):
        super(CQR_logits, self).__init__()

        self.model = model

        self.alpha = alpha
        self.Qhat = conformal_calibration_logits(
            self, calib_loader, len(calib_loader.dataset))

    def forward(self, logits):
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()

            S = [logits_numpy[:, 0] - self.Qhat,
                 logits_numpy[:, 1] + self.Qhat]
        return S


def conformal_calibration_logits(cmodel, calib_loader, n: int):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            targets = targets.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()

            scores = np.maximum(targets - logits[:, 1], logits[:, 0] - targets)
            E = np.concatenate((E, scores))
        Qhat = np.quantile(E, np.ceil(
            (n+1)*(1-cmodel.alpha))/n, interpolation='higher')
        return Qhat
