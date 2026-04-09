"""
Gráfica de impacto del método de backfill sobre todos los modelos.

Lee métricas de metrics/{exp_name}/metrics_ivf_*.json y genera un gráfico de
barras agrupadas donde:
  - Eje X: 8 tipos de modelo con backfill variable (xgb, ridge, sarimax, mlp, gru, lstm, cnngru, rescnngru)
  - Barras agrupadas: 3 métodos de backfill (zero, linear, xgb_backcast)
  - Métricas: MAE, RMSE, R² (3 subplots)
  - Valores promedios sobre las 4 series IVF
  (SARIMA excluido: solo corre baseline sin backfill variable)

Salida: plots/general/backfill_impact.png

Uso:
    python src/plot_backfill_impact.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SERIES = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]

# (nombre_experimento, etiqueta_modelo, método_backfill)
# SARIMA excluido: solo corre una vez (sarima_baseline), no tiene variante por backfill
EXPERIMENT_GRID = {
    "zero": {
        "xgb":        "xgb_zero",
        "ridge":      "ridge_zero",
        "sarimax":    "sarimax_zero",
        "mlp":        "mlp_zero",
        "gru":        "gru_zero",
        "lstm":       "lstm_zero",
        "cnngru":     "cnngru_zero",
        "rescnngru":  "rescnngru_zero",
    },
    "linear": {
        "xgb":        "xgb_linear",
        "ridge":      "ridge_linear",
        "sarimax":    "sarimax_linear",
        "mlp":        "mlp_linear",
        "gru":        "gru_linear",
        "lstm":       "lstm_linear",
        "cnngru":     "cnngru_linear",
        "rescnngru":  "rescnngru_linear",
    },
    "xgb_backcast": {
        "xgb":        "xgb_xgb_backcast",
        "ridge":      "ridge_xgb_backcast",
        "sarimax":    "sarimax_xgb_backcast",
        "mlp":        "mlp_xgb_backcast",
        "gru":        "gru_xgb_backcast",
        "lstm":       "lstm_xgb_backcast",
        "cnngru":     "cnngru_xgb_backcast",
        "rescnngru":  "rescnngru_xgb_backcast",
    },
}

BACKFILL_LABELS = {
    "zero":         "Zero",
    "linear":       "Linear",
    "xgb_backcast": "XGB Backcast",
}

MODEL_LABELS = {
    "xgb":       "XGBoost",
    "ridge":     "Ridge",
    "sarimax":   "SARIMAX",
    "mlp":       "MLP",
    "gru":       "GRU",
    "lstm":      "LSTM",
    "cnngru":    "CNN-GRU",
    "rescnngru": "Res-CNN-GRU",
}

METRICS = ["mae", "rmse", "r2"]
METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "r2": "R²"}
OUT_PATH = "plots/general/backfill_impact.png"


def load_avg_metric(exp_name: str, metric: str) -> float:
    """Carga la métrica promediada sobre las 4 series IVF."""
    vals = []
    for series in SERIES:
        path = f"metrics/{exp_name}/metrics_{series}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if metric in data:
            vals.append(data[metric])
    return float(np.mean(vals)) if vals else float("nan")


def plot() -> None:
    os.makedirs("plots/general", exist_ok=True)

    model_keys = list(MODEL_LABELS.keys())
    backfill_keys = list(BACKFILL_LABELS.keys())
    n_models = len(model_keys)
    n_backfills = len(backfill_keys)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(18, 4 * len(METRICS)))
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_backfills))

    for ax, metric in zip(axes, METRICS):
        x = np.arange(n_models)
        width = 0.7 / n_backfills

        for i, bf_key in enumerate(backfill_keys):
            vals = []
            for model_key in model_keys:
                exp_name = EXPERIMENT_GRID[bf_key][model_key]
                vals.append(load_avg_metric(exp_name, metric))

            bars = ax.bar(
                x + i * width - (n_backfills - 1) * width / 2,
                vals,
                width,
                label=BACKFILL_LABELS[bf_key],
                color=colors[i],
                alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if metric != "r2" else 0.005),
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[k] for k in model_keys])
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(f"{METRIC_LABELS[metric]} por modelo y método de backfill (promedio 4 series)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if metric == "r2":
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_backfills, fontsize=9, frameon=True)
    fig.suptitle(
        "Impacto del método de backfill — métricas en conjunto de prueba (2022 Q1–2025 Q3)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en: {OUT_PATH}")


if __name__ == "__main__":
    plot()
