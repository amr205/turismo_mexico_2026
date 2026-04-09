"""
Modelo Ridge Regression para pronóstico de series de tiempo.

Regresión lineal regularizada con los mismos features que XGBoost.
Muestra la contribución lineal vs. no-lineal (XGBoost) con los mismos inputs.

SHAP via LinearExplainer (exacto y rápido para modelos lineales).
"""

import numpy as np
import joblib
import shap
from sklearn.linear_model import Ridge

from .base import BaseForecaster


class RidgeForecaster(BaseForecaster):
    """Ridge Regression — baseline lineal con mismos features que XGBoost."""

    def __init__(self, params: dict):
        self.params = params
        self.alpha = params.get("alpha", 1.0)
        self.model: Ridge | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        X = np.array(X_train, dtype=np.float64)
        y = np.array(y_train, dtype=np.float64)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        print(f"  Ridge ajustado — alpha={self.alpha}")

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.array(X, dtype=np.float64)).astype(np.float32)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "params": self.params}, path)

    @classmethod
    def load(cls, path: str) -> "RidgeForecaster":
        data = joblib.load(path)
        instance = cls(data["params"])
        instance.model = data["model"]
        return instance

    # ------------------------------------------------------------------
    def get_shap_explainer(self, X_background):
        """LinearExplainer — exacto para modelos lineales."""
        X_bg = np.array(X_background, dtype=np.float64)
        explainer = shap.LinearExplainer(self.model, X_bg)
        shap_values = explainer.shap_values(np.array(X_background, dtype=np.float32))
        return np.array(shap_values, dtype=np.float32)
