import torch
import numpy as np
from tqdm import tqdm

from src.utils.utils import get_lstm_model_std


class ConformalForcast(torch.nn.Module):
    def __init__(self, model, calib_loader, alpha, batch_size=16) -> None:
        super(ConformalForcast, self).__init__()
        self.model = model
        self.alpha = alpha
        self.calib_logits_set = get_logits_targets(self.model, calib_loader)

        self.calib_logit_loader = torch.utils.data.DataLoader(
            self.calib_logits_set, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.Qhat = self.conformal_calibration(self.calib_logit_loader)

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        std = get_lstm_model_std(*args, dropout_fraction=0.3).numpy()
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            S = [logits_numpy - std*self.Qhat,
                 logits_numpy + std*self.Qhat]
        return S

    def conformal_calibration(self, calib_loader):
        n = len(calib_loader.dataset)
        cor_alpha = np.ceil((n+1)*(1-self.alpha))/n

        with torch.no_grad():
            E = torch.tensor([])
            for logits, model_std, targets in calib_loader:
                targets = targets.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
                scores = np.abs(logits - targets)/model_std
                E = torch.cat((E, scores))
            return torch.quantile(E, cor_alpha).item()


def get_logits_targets(model, loader):
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
    logits = torch.zeros((len(loader.dataset),))
    labels = torch.zeros((len(loader.dataset),))
    model_std = torch.zeros((len(loader.dataset),))
    i = 0
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.cpu())
            logits[i:(i+x.shape[0])] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            model_std[i:(i+x.shape[0])] = get_lstm_model_std(x,
                                                             dropout_fraction=0.3)
            i = i + x.shape[0]
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, model_std, labels)

    return dataset_logits
