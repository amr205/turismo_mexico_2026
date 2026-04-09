"""
Gráfica de comparación de modelos entre experimentos archivados.

Lee métricas de metrics/{experiment_name}/metrics_*.json para cada
experimento listado en EXPERIMENTS, y genera un gráfico de barras agrupadas
con MAE, RMSE y R² por serie.

Salida: plots/general/model_comparison.png

Uso (después de archivar todos los experimentos):
    python src/plot_model_comparison.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Experimentos a comparar (nombre → etiqueta en la gráfica)
EXPERIMENTS = {
    "xgb_linear":       "XGBoost\n(linear)",
    "ridge_linear":     "Ridge\n(linear)",
    "sarima_baseline":  "SARIMA\n(baseline)",
    "sarimax_linear":   "SARIMAX\n(linear)",
    "mlp_linear":       "MLP\n(linear)",
    "gru_linear":       "GRU\n(linear)",
    "lstm_linear":      "LSTM\n(linear)",
    "cnngru_linear":    "CNN-GRU\n(linear)",
    "rescnngru_linear": "Res-CNN-GRU\n(linear)",
}

SERIES = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]

SERIES_LABELS = {
    "ivf_total_nacional": "Total Nacional",
    "ivf_turistico_total": "Turístico Total",
    "ivf_turistico_bienes": "Turístico Bienes",
    "ivf_turistico_servicios": "Turístico Servicios",
}

METRICS = ["mae", "rmse", "r2"]
METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "r2": "R²"}
OUT_PATH = "plots/general/model_comparison.png"


def load_metrics(exp_name: str, series: str) -> dict | None:
    path = f"metrics/{exp_name}/metrics_{series}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot() -> None:
    os.makedirs("plots/general", exist_ok=True)

    exp_names = list(EXPERIMENTS.keys())
    exp_labels = list(EXPERIMENTS.values())
    n_exp = len(exp_names)
    n_series = len(SERIES)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(20, 4 * len(METRICS)))
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_exp))

    for ax, metric in zip(axes, METRICS):
        x = np.arange(n_series)
        width = 0.8 / n_exp
        for i, (exp, label) in enumerate(zip(exp_names, exp_labels)):
            vals = []
            for s in SERIES:
                m = load_metrics(exp, s)
                vals.append(m[metric] if m and metric in m else float("nan"))
            bars = ax.bar(
                x + i * width - (n_exp - 1) * width / 2,
                vals,
                width,
                label=label,
                color=colors[i],
                alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if metric != "r2" else 0.01),
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([SERIES_LABELS[s] for s in SERIES])
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(f"{METRIC_LABELS[metric]} por serie y modelo", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if metric == "r2":
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_exp, fontsize=9, frameon=True)
    fig.suptitle("Comparación de modelos — métricas en conjunto de prueba (2022 Q1–2025 Q3)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en: {OUT_PATH}")


if __name__ == "__main__":
    plot()
