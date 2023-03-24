import numpy as np


from sklearn.ensemble import GradientBoostingRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Iterable, List, Optional, Tuple, Union, cast
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)
from sklearn.pipeline import Pipeline
from numpy.typing import ArrayLike


class GB_quantile_regressor:
    def __init__(self, low=GradientBoostingRegressor(alpha=0.05, loss='quantile'), up=GradientBoostingRegressor(alpha=0.95, loss='quantile'), median=GradientBoostingRegressor(alpha=0.5, loss='quantile')):
        self.low = low
        self.up = up
        self.median = median

    def predict(self, inputs):
        y_pred_low = self.low.predict(inputs).reshape(-1, 1)
        y_pred_up = self.up.predict(inputs).reshape(-1, 1)
        y_pred_median = self.median.predict(inputs)
        return y_pred_median, np.stack([y_pred_low, y_pred_up], axis=1)


class ConformalizQuantileRegressor(RegressorMixin, BaseEstimator):
    def __init__(
            self,
            alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def fit_precomputed_logits(self, calib_loader):
        n = len(calib_loader.dataset)
        q = (1 - (self.alpha)) * (1 + (1 / n))
        E = np.array([])
        for logits, targets in calib_loader:
            targets = targets.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            scores = np.maximum(
                targets - logits[:, 1], logits[:, 0] - targets)
            E = np.concatenate((E, scores))
        self.Qhat = np.quantile(E, q, interpolation='higher')
        return self

    def predict(self, X: ArrayLike):
        X = X.detach().cpu().numpy()
        S = [X[:, 0] - self.Qhat,
             X[:, 1] + self.Qhat]
        return S
