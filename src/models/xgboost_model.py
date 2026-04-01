import numpy as np
import shap
import xgboost as xgb

from .base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """Modelo de pronóstico basado en XGBoost."""

    def __init__(self, params: dict):
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "XGBoostForecaster":
        instance = cls.__new__(cls)
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(path)
        return instance

    def get_shap_explainer(self, X_background):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_background)
        return shap_values
