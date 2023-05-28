import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostQunatileRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        low=None,
        up=None,
        median=None,
    ):
        self.low = low
        self.up = up
        self.median = median

    def predict(self, inputs):
        y_pred_low = self.low.predict(inputs.numpy()).reshape(-1, 1)
        y_pred_up = self.up.predict(inputs.numpy()).reshape(-1, 1)
        return np.concatenate([y_pred_low, y_pred_up], axis=1)
