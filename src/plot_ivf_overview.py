"""
Gráfica general de todas las series IVF con banda sombreada de COVID-19.

Salida: plots/general/ivf_overview.png

Uso:
    python src/plot_ivf_overview.py
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

IVF_PATH = "data/processed/indice_volumen_fisico_inegi_clean.csv"
OUT_PATH = "plots/general/ivf_overview.png"

SERIES = {
    "ivf_total_nacional": "IVF Total Nacional",
    "ivf_turistico_total": "IVF Turístico Total",
    "ivf_turistico_bienes": "IVF Turístico Bienes",
    "ivf_turistico_servicios": "IVF Turístico Servicios",
}

COVID_START = pd.Timestamp("2020-01-01")
COVID_END = pd.Timestamp("2021-10-01")


def plot() -> None:
    df = pd.read_csv(IVF_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    os.makedirs("plots/general", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, SERIES.items()):
        if col not in df.columns:
            ax.set_title(f"{label}\n(serie no encontrada)")
            continue
        serie = df[col]
        ax.plot(serie.index, serie.values, color="steelblue", linewidth=1.5)
        ax.axvspan(COVID_START, COVID_END, color="lightcoral", alpha=0.35, label="COVID-19")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Índice", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    # Leyenda única en la primera subgráfica
    covid_patch = mpatches.Patch(color="lightcoral", alpha=0.5, label="COVID-19 (2020 Q1–2021 Q4)")
    fig.legend(handles=[covid_patch], loc="lower center", ncol=1, fontsize=10, frameon=True)

    fig.suptitle("Índice de Volumen Físico del Turismo — INEGI", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en: {OUT_PATH}")


if __name__ == "__main__":
    plot()
