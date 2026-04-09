"""
Modelo SARIMA para pronóstico de series de tiempo.

Baseline clásico de series de tiempo sin variables exógenas.
Usa statsmodels SARIMAX sin exog: order=(1,1,1), seasonal_order=(1,1,1,4).

SHAP no aplica — get_shap_explainer retorna ceros.
interpret.py omite el plot SHAP cuando todos los valores son cero.
"""

import pickle

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    """SARIMA puro — baseline temporal sin indicadores externos."""

    def __init__(self, params: dict):
        self.params = params
        self.order = tuple(params.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(params.get("seasonal_order", [1, 1, 1, 4]))
        self.results = None
        self.n_train: int = 0

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        # Ignora X_train — SARIMA solo usa la serie temporal
        y = np.array(y_train, dtype=np.float64)
        self.n_train = len(y)
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results = model.fit(disp=False)
        print(f"  SARIMA ajustado — AIC={self.results.aic:.2f}")

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        n = len(X)
        forecast = self.results.get_forecast(steps=n)
        return np.asarray(forecast.predicted_mean).astype(np.float32)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "results": self.results,
                    "n_train": self.n_train,
                    "params": self.params,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "SARIMAForecaster":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(data["params"])
        instance.results = data["results"]
        instance.n_train = data["n_train"]
        return instance

    # ------------------------------------------------------------------
    def get_shap_explainer(self, X_background):
        """SARIMA no tiene importancia de features — retorna matriz de ceros."""
        X_bg = np.array(X_background)
        return np.zeros((X_bg.shape[0], X_bg.shape[1]), dtype=np.float32)
