"""
Modelo Res-CNN-GRU para pronóstico de series de tiempo.

Arquitectura completa (basada en imagen de referencia, adaptada a regresión):
    5 ResConvBlock (Conv1D × 2 + BN + conexión residual + MaxPool)
      canales: n_features → 32 → 64 → 128 → 256 → 256
      seq_len: 16 → 15 → 14 → 13 → 12 → 11
    3 capas GRU (hidden=256)
    FC(256 → 256) → ReLU → Dropout
    FC(256 → 128) → ReLU → Dropout
    FC(128 → 1)

Evalúa la contribución de las conexiones residuales y la jerarquía de características
más profunda frente al CNN-GRU sin residuales.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseForecaster


class _ResConvBlock(nn.Module):
    """
    Bloque residual para Conv1D.
    Si in_ch != out_ch usa una convolución 1×1 en el atajo (shortcut).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        # Atajo de proyección cuando los canales cambian
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return self.pool(out)


class _ResCNNGRUNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: list,
        gru_hidden: int,
        gru_layers: int,
        fc_layers: list,
        dropout: float,
    ):
        super().__init__()
        # Bloques residuales
        res_blocks = []
        in_ch = n_features
        for out_ch in conv_channels:
            res_blocks.append(_ResConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.res_blocks = nn.ModuleList(res_blocks)

        # GRU (apila num_layers manualmente para dropout uniforme entre capas)
        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Capas FC
        fc = []
        in_fc = gru_hidden
        for h in fc_layers:
            fc += [nn.Linear(in_fc, h), nn.ReLU(), nn.Dropout(dropout)]
            in_fc = h
        fc.append(nn.Linear(in_fc, 1))
        self.fc = nn.Sequential(*fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)          # → (batch, n_features, seq_len)
        for block in self.res_blocks:
            x = block(x)                 # → (batch, out_ch, seq_len')
        x = x.permute(0, 2, 1)          # → (batch, seq_len', out_ch)
        _, h_n = self.gru(x)             # h_n: (layers, batch, hidden)
        out = h_n[-1]                    # (batch, hidden)
        return self.fc(out).squeeze(-1)


class ResCNNGRUForecaster(BaseForecaster):
    """Res-CNN-GRU — arquitectura completa con conexiones residuales y GRU profundo."""

    def __init__(self, params: dict):
        self.params = params
        self.seq_len = params.get("sequence_length", 16)
        self.conv_channels = params.get("conv_channels", [32, 64, 128, 256, 256])
        self.gru_hidden = params.get("gru_hidden_size", 256)
        self.gru_layers = params.get("gru_num_layers", 3)
        self.fc_layers = params.get("fc_layers", [256, 128])
        self.dropout = params.get("dropout", 0.3)
        self.epochs = params.get("epochs", 200)
        self.lr = params.get("learning_rate", 0.001)
        self.batch_size = params.get("batch_size", 16)
        self.model: _ResCNNGRUNet | None = None
        self.n_features: int | None = None
        self.context_X: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> None:
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)
        self.n_features = X.shape[1]
        self.context_X = X[-(self.seq_len - 1):]

        self.model = _ResCNNGRUNet(
            self.n_features,
            self.conv_channels,
            self.gru_hidden,
            self.gru_layers,
            self.fc_layers,
            self.dropout,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

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
                print(f"  Res-CNN-GRU época {epoch + 1}/{self.epochs}")

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
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ResCNNGRUForecaster":
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["params"])
        instance.n_features = data["n_features"]
        instance.context_X = data["context_X"]
        instance.model = _ResCNNGRUNet(
            instance.n_features,
            instance.conv_channels,
            instance.gru_hidden,
            instance.gru_layers,
            instance.fc_layers,
            instance.dropout,
        )
        instance.model.load_state_dict(data["state_dict"])
        instance.model.eval()
        return instance

    # ------------------------------------------------------------------
    def get_shap_explainer(self, X_background):
        import shap

        X_bg = np.array(X_background, dtype=np.float32)
        full = np.concatenate([self.context_X, X_bg], axis=0)
        n = len(X_bg)
        X_wins = np.stack(
            [full[i : i + self.seq_len] for i in range(n)], axis=0
        )
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
        # (n, seq_len, n_features) → promedio temporal → (n, n_features)
        return shap_values.mean(axis=1)
