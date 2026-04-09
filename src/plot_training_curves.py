"""
Gráficas de curvas de entrenamiento (train vs val loss) para modelos NN.

Lee models/{exp_name}/model_{series}.json (archivos PyTorch con train_losses/val_losses)
y genera una figura por tipo de modelo con subplots: filas = series IVF, columnas = backfills.

Salida: plots/general/training_curves_{model_type}.png

Uso:
    python src/plot_training_curves.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

MODEL_TYPES = ["mlp", "gru", "lstm", "cnn_gru", "res_cnn_gru"]
SERIES = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]
SERIES_LABELS = {
    "ivf_total_nacional":    "Total Nacional",
    "ivf_turistico_total":   "Turístico Total",
    "ivf_turistico_bienes":  "Turístico Bienes",
    "ivf_turistico_servicios": "Turístico Servicios",
}
BACKFILLS = ["zero", "linear", "xgb_backcast"]
BACKFILL_LABELS = {
    "zero":        "Zero",
    "linear":      "Linear",
    "xgb_backcast":"XGB Backcast",
}

# Nombre de experimento por model_type × backfill
EXP_NAMES = {
    "mlp":        {"zero": "mlp_zero",       "linear": "mlp_linear",       "xgb_backcast": "mlp_xgb_backcast"},
    "gru":        {"zero": "gru_zero",       "linear": "gru_linear",       "xgb_backcast": "gru_xgb_backcast"},
    "lstm":       {"zero": "lstm_zero",      "linear": "lstm_linear",      "xgb_backcast": "lstm_xgb_backcast"},
    "cnn_gru":    {"zero": "cnngru_zero",    "linear": "cnngru_linear",    "xgb_backcast": "cnngru_xgb_backcast"},
    "res_cnn_gru":{"zero": "rescnngru_zero", "linear": "rescnngru_linear", "xgb_backcast": "rescnngru_xgb_backcast"},
}

MODEL_DISPLAY = {
    "mlp":        "MLP",
    "gru":        "GRU",
    "lstm":       "LSTM",
    "cnn_gru":    "CNN-GRU",
    "res_cnn_gru":"Res-CNN-GRU",
}


def load_losses(exp_name: str, series: str) -> tuple[list, list]:
    """Carga train_losses y val_losses del archivo del modelo archivado."""
    path = f"models/{exp_name}/model_{series}.json"
    if not os.path.exists(path):
        return [], []
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        train = data.get("train_losses", [])
        val = data.get("val_losses", [])
        return train, val
    except Exception as e:
        print(f"  WARN: no se pudo cargar {path}: {e}")
        return [], []


def plot_model(model_type: str) -> None:
    n_series = len(SERIES)
    n_backfills = len(BACKFILLS)
    fig, axes = plt.subplots(
        n_series, n_backfills,
        figsize=(5 * n_backfills, 4 * n_series),
        squeeze=False,
    )

    for row, series in enumerate(SERIES):
        for col, bf in enumerate(BACKFILLS):
            ax = axes[row][col]
            exp_name = EXP_NAMES[model_type][bf]
            train_losses, val_losses = load_losses(exp_name, series)

            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                ax.plot(epochs, train_losses, label="Train", color="steelblue", linewidth=1.5)
                if val_losses:
                    ax.plot(epochs, val_losses, label="Val", color="darkorange",
                            linewidth=1.5, linestyle="--")
                ax.set_xlabel("Época", fontsize=8)
                ax.set_ylabel("MSE Loss", fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_axis_off()

            title = f"{SERIES_LABELS[series]}\n({BACKFILL_LABELS[bf]})"
            ax.set_title(title, fontsize=9)

    model_display = MODEL_DISPLAY[model_type]
    fig.suptitle(f"Curvas de entrenamiento — {model_display}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs("plots/general", exist_ok=True)
    out_path = f"plots/general/training_curves_{model_type}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en: {out_path}")


def main() -> None:
    for model_type in MODEL_TYPES:
        plot_model(model_type)


if __name__ == "__main__":
    main()
