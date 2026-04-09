"""
Modelo SARIMAX para pronóstico de series de tiempo.

SARIMA con variables exógenas (indicadores de turismo INEGI).
Muestra el enfoque clásico con features externos vs. SARIMA puro.

SHAP via KernelExplainer (lento — background reducido a 20 muestras).
"""

import pickle

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster


class SARIMAXForecaster(BaseForecaster):
    """SARIMAX con indicadores exógenos — SARIMA + features de turismo."""

    def __init__(self, params: dict):
        self.params = params
        self.order = tuple(params.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(params.get("seasonal_order", [1, 1, 1, 4]))
        self.results = None
        self.n_train: int = 0

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        y = np.array(y_train, dtype=np.float64)
        X = np.array(X_train, dtype=np.float64)
        self.n_train = len(y)
        model = SARIMAX(
            y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results = model.fit(disp=False)
        print(f"  SARIMAX ajustado — AIC={self.results.aic:.2f}")

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X_exog = np.array(X, dtype=np.float64)
        n = len(X_exog)
        forecast = self.results.get_forecast(steps=n, exog=X_exog)
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
    def load(cls, path: str) -> "SARIMAXForecaster":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(data["params"])
        instance.results = data["results"]
        instance.n_train = data["n_train"]
        return instance

    # ------------------------------------------------------------------
    def get_shap_explainer(self, X_background):
        """KernelExplainer sobre predict() — background reducido a 20 muestras."""
        import shap

        X_bg = np.array(X_background, dtype=np.float32)
        background = shap.sample(X_bg, min(20, len(X_bg)))

        def pred_fn(x):
            return self.predict(np.array(x, dtype=np.float64))

        explainer = shap.KernelExplainer(pred_fn, background)
        shap_values = explainer.shap_values(X_bg)
        return np.array(shap_values, dtype=np.float32)
