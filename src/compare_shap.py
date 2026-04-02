"""
Comparación cruzada de importancia de indicadores de turismo en las 4 series IVF.

Carga los valores SHAP persistidos por interpret.py y genera:
  - Heatmap: filas = indicadores, columnas = series IVF, color = importancia normalizada.
  - CSV de importancia para tablas del paper.

Uso:
    python src/compare_shap.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SERIES = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]

# Features autorregresivas que se excluyen del análisis de indicadores
BASE_PREFIXES = ("lag_", "roll_mean_", "q2", "q3", "q4", "trend")


def es_feature_base(nombre: str) -> bool:
    return any(nombre.startswith(p) or nombre == p for p in BASE_PREFIXES)


def cargar_importancia(series: str) -> pd.Series:
    """Carga shap_values y devuelve importancia media absoluta por feature."""
    npz_path = f"data/shap/shap_values_{series}.npz"
    json_path = f"data/shap/feature_names_{series}.json"

    data = np.load(npz_path)
    shap_values = data["shap_values"]

    with open(json_path) as f:
        feature_names = json.load(f)

    importancia = np.abs(shap_values).mean(axis=0)
    return pd.Series(importancia, index=feature_names, name=series)


def main() -> None:
    # --- Cargar importancia de todas las series ---
    series_imp = []
    for s in SERIES:
        npz = f"data/shap/shap_values_{s}.npz"
        if not os.path.exists(npz):
            raise FileNotFoundError(
                f"No se encontró {npz}. Ejecuta primero: dvc repro interpret_ivf_multi"
            )
        series_imp.append(cargar_importancia(s))

    df_imp = pd.concat(series_imp, axis=1)

    # --- Filtrar sólo columnas de indicadores ---
    mask_ind = ~pd.Index(df_imp.index).to_series().apply(es_feature_base)
    df_ind = df_imp.loc[mask_ind].copy()

    if df_ind.empty:
        print("No se encontraron columnas de indicadores en los valores SHAP.")
        return

    # Ordenar indicadores por importancia total absoluta
    df_ind = df_ind.loc[df_ind.sum(axis=1).sort_values(ascending=False).index]

    # Identificar qué indicadores tienen al menos un valor no nulo
    mask_nonzero = df_ind.sum(axis=1) > 0
    n_nonzero = mask_nonzero.sum()

    print(f"Indicadores con SHAP > 0 en alguna serie: {n_nonzero} / {len(df_ind)}")
    if n_nonzero == 0:
        print("Advertencia: todos los indicadores tienen SHAP=0. Los indicadores no aportan señal con la configuración actual.")

    # Usar escala log para el heatmap (evita que valores ~0 oculten diferencias reales)
    # Añadir epsilon para evitar log(0)
    eps = df_ind[df_ind > 0].min().min() * 0.01 if n_nonzero > 0 else 1e-10
    df_log = np.log10(df_ind + eps)

    # --- Heatmap en escala log ---
    os.makedirs("plots", exist_ok=True)
    fig_h = max(8, len(df_ind) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    # Anotaciones: mostrar valor absoluto real como string (evita bugs de seaborn con fmt="")
    annot = df_ind.map(lambda x: f"{x:.5f}" if x > 1e-9 else "0.0")
    sns.heatmap(
        df_log,
        ax=ax,
        cmap="YlOrRd",
        annot=annot,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "log₁₀(media |SHAP|)"},
    )
    ax.set_title(
        "Importancia de indicadores de turismo por serie IVF\n"
        "(media |SHAP| absoluta — escala log₁₀)\n"
        "Valores = 0.00000 indican que XGBoost no usó ese indicador"
    )
    ax.set_xlabel("Serie IVF")
    ax.set_ylabel("Indicador")
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out_heatmap = "plots/shap_compare_indicators.png"
    fig.savefig(out_heatmap, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Heatmap guardado en: {out_heatmap}")

    # --- CSV con importancia absoluta (sin normalizar) para el paper ---
    os.makedirs("data/shap", exist_ok=True)
    out_csv = "data/shap/indicator_importance_summary.csv"
    df_ind.round(6).to_csv(out_csv)
    print(f"Tabla de importancia guardada en: {out_csv}")
    print(f"\nTop 5 indicadores por importancia total:")
    print(df_ind.sum(axis=1).sort_values(ascending=False).head(5).to_string())


if __name__ == "__main__":
    main()
