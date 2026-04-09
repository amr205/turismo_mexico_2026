"""
Modelo LSTM para pronóstico de series de tiempo.

Arquitectura:
    LSTM(n_features, hidden=64, layers=1) → último estado oculto h_n[-1]
    FC(64 → 32) → ReLU → Dropout
    FC(32 → 1)

Companion natural del GRU — misma interfaz, nn.LSTM en lugar de nn.GRU.
Incluye val split (20%) + early stopping con patience=30.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from .base import BaseForecaster


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden)
        out = h_n[-1]                 # último estado oculto: (batch, hidden)
        return self.fc(out).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """LSTM con ventanas deslizantes — companion del GRU con celda de memoria explícita."""

    def __init__(self, params: dict):
        self.params = params
        self.hidden_size = params.get("hidden_size", 64)
        self.num_layers = params.get("num_layers", 1)
        self.seq_len = params.get("sequence_length", 8)
        self.dropout = params.get("dropout", 0.2)
        self.epochs = params.get("epochs", 500)
        self.lr = params.get("learning_rate", 0.0005)
        self.batch_size = params.get("batch_size", 8)
        self.model: _LSTMNet | None = None
        self.n_features: int | None = None
        self.context_X: np.ndarray | None = None   # últimas seq_len-1 filas de entrenamiento
        self.train_losses: list = []
        self.val_losses: list = []

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)
        self.n_features = X.shape[1]
        self.context_X = X[-(self.seq_len - 1):]

        self.model = _LSTMNet(self.n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Ventanas deslizantes
        n = len(X)
        X_wins = np.stack([X[i : i + self.seq_len] for i in range(n - self.seq_len)], axis=0)
        y_wins = y[self.seq_len:]

        # Val split cronológico (último 20%)
        n_val = max(1, int(len(X_wins) * 0.2))
        X_tr, X_vl = X_wins[:-n_val], X_wins[-n_val:]
        y_tr, y_vl = y_wins[:-n_val], y_wins[-n_val:]

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
                    print(f"  LSTM early stop época {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=np.float32)
        full = np.concatenate([self.context_X, X], axis=0)
        n = len(X)
        X_wins = np.stack(
            [full[i : i + self.seq_len] for i in range(n)], axis=0
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X_wins)).numpy()

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "context_X": self.context_X,
                "params": self.params,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMForecaster":
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["params"])
        instance.n_features = data["n_features"]
        instance.context_X = data["context_X"]
        instance.train_losses = data.get("train_losses", [])
        instance.val_losses = data.get("val_losses", [])
        instance.model = _LSTMNet(
            instance.n_features,
            instance.hidden_size,
            instance.num_layers,
            instance.dropout,
        )
        instance.model.load_state_dict(data["state_dict"])
        instance.model.eval()
        return instance

    # ------------------------------------------------------------------
    def get_shap_explainer(self, X_background):
        """SHAP via GradientExplainer. Valores promediados sobre la dimensión temporal."""
        import shap

        X_bg = np.array(X_background, dtype=np.float32)
        full = np.concatenate([self.context_X, X_bg], axis=0)
        n = len(X_bg)
        X_wins = np.stack(
            [full[i : i + self.seq_len] for i in range(n)], axis=0
        )  # (n, seq_len, n_features)
        X_t = torch.from_numpy(X_wins)

        self.model.eval()

        class _Wrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x).unsqueeze(-1)

        explainer = shap.GradientExplainer(_Wrap(self.model), X_t)
        shap_values = explainer.shap_values(X_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        # GradientExplainer devuelve (n, seq_len, n_features, 1) con wrapper → eliminar última dim
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        # Promediar sobre la dimensión de secuencia: (n, seq_len, n_features) → (n, n_features)
        return shap_values.mean(axis=1)
