"""
Modelo GRU para pronóstico de series de tiempo.

Arquitectura:
    GRU(n_features, hidden=128, layers=2) → último estado oculto
    FC(128 → 64) → ReLU → Dropout
    FC(64 → 1)

Usa ventanas deslizantes de seq_len pasos como entrada.
Aísla la contribución de la recurrencia temporal frente al MLP.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseForecaster


class _GRUNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
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
        _, h_n = self.gru(x)   # h_n: (num_layers, batch, hidden)
        out = h_n[-1]          # último estado oculto: (batch, hidden)
        return self.fc(out).squeeze(-1)


class GRUForecaster(BaseForecaster):
    """GRU con ventanas deslizantes — agrega recurrencia temporal sobre el MLP."""

    def __init__(self, params: dict):
        self.params = params
        self.hidden_size = params.get("hidden_size", 128)
        self.num_layers = params.get("num_layers", 2)
        self.seq_len = params.get("sequence_length", 16)
        self.dropout = params.get("dropout", 0.3)
        self.epochs = params.get("epochs", 200)
        self.lr = params.get("learning_rate", 0.001)
        self.batch_size = params.get("batch_size", 16)
        self.model: _GRUNet | None = None
        self.n_features: int | None = None
        self.context_X: np.ndarray | None = None   # últimas seq_len-1 filas de entrenamiento

    # ------------------------------------------------------------------
    def _make_sequences(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crea pares (ventana de seq_len pasos, índice del paso a predecir)."""
        n = len(X)
        xs = np.stack([X[i : i + self.seq_len] for i in range(n - self.seq_len)], axis=0)
        # targets en posición seq_len (primer valor fuera de la ventana)
        return xs  # shape: (n - seq_len, seq_len, n_features)

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)
        self.n_features = X.shape[1]
        self.context_X = X[-(self.seq_len - 1):]  # contexto para predicción en test

        self.model = _GRUNet(self.n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Ventanas: predecir y[i + seq_len] desde X[i : i + seq_len]
        n = len(X)
        X_wins = np.stack([X[i : i + self.seq_len] for i in range(n - self.seq_len)], axis=0)
        y_wins = y[self.seq_len:]

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_wins), torch.from_numpy(y_wins)
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
                print(f"  GRU época {epoch + 1}/{self.epochs}")

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=np.float32)
        # Anteponer contexto de entrenamiento para evitar información del futuro
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
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "GRUForecaster":
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["params"])
        instance.n_features = data["n_features"]
        instance.context_X = data["context_X"]
        instance.model = _GRUNet(
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
