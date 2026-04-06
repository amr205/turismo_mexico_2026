"""
Interpretación del modelo: SHAP y descomposición STL.

Uso:
    python src/interpret.py --dataset consumo
    python src/interpret.py --dataset ivf
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models import get_model_class

DATASET_CONFIG = {
    "consumo": {
        "features": "data/features/features_consumo.csv",
        "processed": "data/processed/consumo_turistico_inegi_clean.csv",
        "target_key": "consumo",
        "split_meta": "models/split_consumo.json",
        "shap_out": "plots/shap_consumo.png",
        "stl_out": "plots/stl_consumo.png",
    },
    "ivf": {
        "features": "data/features/features_ivf.csv",
        "processed": "data/processed/indice_volumen_fisico_inegi_clean.csv",
        "target_key": "ivf",
        "split_meta": "models/split_ivf.json",
        "shap_out": "plots/shap_ivf.png",
        "stl_out": "plots/stl_ivf.png",
    },
}


def resolver_dataset(dataset: str, params: dict) -> tuple:
    """Devuelve (cfg, target_col) para datasets legacy o ivf_multi."""
    if dataset in DATASET_CONFIG:
        cfg = DATASET_CONFIG[dataset]
        target_col = params["target"][cfg["target_key"]]
        return cfg, target_col
    target_col = params["ivf_multi"][dataset]["target"]
    cfg = {
        "features": f"data/features/features_{dataset}.csv",
        "processed": "data/processed/indice_volumen_fisico_inegi_clean.csv",
        "split_meta": f"models/split_{dataset}.json",
        "shap_out": f"plots/shap_{dataset}.png",
        "stl_out": f"plots/stl_{dataset}.png",
    }
    return cfg, target_col


def interpret(dataset: str) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    cfg, target_col = resolver_dataset(dataset, params)
    model_type = params["model"]["type"]
    stl_period = params["interpret"]["stl_period"]
    shap_max_display = params["interpret"]["shap_max_display"]

    with open(cfg["split_meta"]) as f:
        split = json.load(f)
    n_train = split["n_train"]

    # --- SHAP ---
    df_feat = pd.read_csv(cfg["features"], parse_dates=["date"])
    df_feat = df_feat.set_index("date").sort_index()
    X = df_feat.drop(columns=[target_col])
    X_train = X.iloc[:n_train]

    ModelClass = get_model_class(model_type)
    model = ModelClass.load(f"models/model_{dataset}.json")

    shap_values = model.get_shap_explainer(X_train)

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        max_display=shap_max_display,
        show=False,
    )
    plt.title(f"Importancia de características (SHAP) — {dataset}")
    plt.tight_layout()
    plt.savefig(cfg["shap_out"], bbox_inches="tight")
    plt.close()
    print(f"Gráfica SHAP guardada en: {cfg['shap_out']}")

    # Persistir valores SHAP y nombres de features para análisis cruzado
    if params.get("interpret", {}).get("save_shap_values", False):
        os.makedirs("data/shap", exist_ok=True)
        shap_npz = f"data/shap/shap_values_{dataset}.npz"
        feat_json = f"data/shap/feature_names_{dataset}.json"
        np.savez(shap_npz, shap_values=shap_values)
        with open(feat_json, "w") as f:
            json.dump(list(X_train.columns), f)
        print(f"Valores SHAP guardados en: {shap_npz}")

    # --- STL ---
    df_proc = pd.read_csv(cfg["processed"], parse_dates=["date"])
    df_proc = df_proc.set_index("date").sort_index()
    series = df_proc[target_col].dropna()

    # STL requiere al menos 2*period+1 observaciones
    stl = STL(series, period=stl_period).fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(series.index, stl.observed, color="steelblue")
    axes[0].set_title("Observado")
    axes[1].plot(series.index, stl.trend, color="darkorange")
    axes[1].set_title("Tendencia")
    axes[2].plot(series.index, stl.seasonal, color="green")
    axes[2].set_title("Estacionalidad")
    axes[3].plot(series.index, stl.resid, color="gray")
    axes[3].set_title("Residuo")
    for ax in axes:
        ax.grid(True)
    fig.suptitle(f"Descomposición STL — {dataset}", fontsize=13)
    fig.tight_layout()
    fig.savefig(cfg["stl_out"], bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfica STL guardada en: {cfg['stl_out']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    interpret(args.dataset)
