"""
Modelo MLP (Perceptrón Multicapa) para pronóstico de series de tiempo.

Arquitectura:
    FC(n_features → 128) → BN → ReLU → Dropout
    FC(128 → 64)          → BN → ReLU → Dropout
    FC(64 → 1)

Sin estructura temporal — entrada tabular idéntica a XGBoost.
Sirve como línea base de red neuronal para comparar con los modelos recurrentes.
"""

import json

import numpy as np
import torch
import torch.nn as nn

from .base import BaseForecaster


class _MLPNet(nn.Module):
    def __init__(self, n_features: int, hidden_layers: list, dropout: float):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_layers:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPForecaster(BaseForecaster):
    """Perceptrón multicapa — comparación tabular directa con XGBoost."""

    def __init__(self, params: dict):
        self.params = params
        self.hidden_layers = params.get("hidden_layers", [128, 64])
        self.dropout = params.get("dropout", 0.3)
        self.epochs = params.get("epochs", 200)
        self.lr = params.get("learning_rate", 0.001)
        self.batch_size = params.get("batch_size", 16)
        self.model: _MLPNet | None = None
        self.n_features: int | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)
        self.n_features = X.shape[1]

        self.model = _MLPNet(self.n_features, self.hidden_layers, self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"  MLP época {epoch + 1}/{self.epochs}")

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X_t = torch.from_numpy(np.array(X, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).numpy()

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MLPForecaster":
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["params"])
        instance.n_features = data["n_features"]
        instance.model = _MLPNet(
            instance.n_features, instance.hidden_layers, instance.dropout
        )
        instance.model.load_state_dict(data["state_dict"])
        instance.model.eval()
        return instance

    # ------------------------------------------------------------------
    def get_shap_explainer(self, X_background):
        import shap

        X_bg = torch.from_numpy(np.array(X_background, dtype=np.float32))
        self.model.eval()

        class _Wrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x).unsqueeze(-1)

        explainer = shap.GradientExplainer(_Wrap(self.model), X_bg)
        shap_values = explainer.shap_values(X_bg)
        # shap_values puede ser lista (una entrada → primer elemento)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        # GradientExplainer devuelve (n, n_features, 1) con wrapper → eliminar última dim
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        return shap_values
