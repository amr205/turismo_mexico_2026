from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """Interfaz que deben implementar todos los modelos de pronóstico."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entrena el modelo con datos de entrenamiento."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones para un conjunto de datos."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Guarda el modelo serializado en disco."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseForecaster":
        """Carga un modelo serializado desde disco."""

    @abstractmethod
    def get_shap_explainer(self, X_background):
        """
        Retorna (shap_values, feature_names) calculados sobre X_background.

        Cada subclase elige el explainer adecuado:
          - XGBoost  → shap.TreeExplainer  (exacto, rápido)
          - LSTM/NN  → shap.DeepExplainer o shap.KernelExplainer (aproximado)
        """
