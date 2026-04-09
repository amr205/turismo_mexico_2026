"""
Modelo MLP (Perceptrón Multicapa) para pronóstico de series de tiempo.

Arquitectura:
    FC(n_features → 64) → BN → ReLU → Dropout
    FC(64 → 32)          → BN → ReLU → Dropout
    FC(32 → 1)

Sin estructura temporal — entrada tabular idéntica a XGBoost.
Sirve como línea base de red neuronal para comparar con los modelos recurrentes.
Incluye val split (20%) + early stopping con patience=30.
"""

import copy

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
        self.hidden_layers = params.get("hidden_layers", [64, 32])
        self.dropout = params.get("dropout", 0.2)
        self.epochs = params.get("epochs", 500)
        self.lr = params.get("learning_rate", 0.0005)
        self.batch_size = params.get("batch_size", 8)
        self.model: _MLPNet | None = None
        self.n_features: int | None = None
        self.train_losses: list = []
        self.val_losses: list = []

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)
        self.n_features = X.shape[1]

        # Val split cronológico (último 20%)
        n_val = max(1, int(len(X) * 0.2))
        X_tr, X_vl = X[:-n_val], X[-n_val:]
        y_tr, y_vl = y[:-n_val], y[-n_val:]

        self.model = _MLPNet(self.n_features, self.hidden_layers, self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        patience = self.params.get("early_stopping_patience", 30)
        self.train_losses, self.val_losses = [], []
        best_val = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    self.model(torch.from_numpy(X_vl)),
                    torch.from_numpy(y_vl),
                ).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                wait = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    print(f"  MLP early stop época {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

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
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MLPForecaster":
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["params"])
        instance.n_features = data["n_features"]
        instance.train_losses = data.get("train_losses", [])
        instance.val_losses = data.get("val_losses", [])
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
