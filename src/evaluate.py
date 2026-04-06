"""
Evaluación del modelo entrenado: métricas y gráfica de pronóstico.

Uso:
    python src/evaluate.py --dataset consumo
    python src/evaluate.py --dataset ivf
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models import get_model_class

DATASET_CONFIG = {
    "consumo": {
        "features": "data/features/features_consumo.csv",
        "target_key": "consumo",
        "split_meta": "models/split_consumo.json",
        "metrics_out": "metrics/metrics_consumo.json",
        "plot_out": "plots/forecast_consumo.png",
    },
    "ivf": {
        "features": "data/features/features_ivf.csv",
        "target_key": "ivf",
        "split_meta": "models/split_ivf.json",
        "metrics_out": "metrics/metrics_ivf.json",
        "plot_out": "plots/forecast_ivf.png",
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
        "split_meta": f"models/split_{dataset}.json",
        "metrics_out": f"metrics/metrics_{dataset}.json",
        "plot_out": f"plots/forecast_{dataset}.png",
    }
    return cfg, target_col


def evaluate(dataset: str) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    cfg, target_col = resolver_dataset(dataset, params)
    model_type = params["model"]["type"]

    with open(cfg["split_meta"]) as f:
        split = json.load(f)

    n_train = split["n_train"]

    df = pd.read_csv(cfg["features"], parse_dates=["date"])
    df = df.set_index("date").sort_index()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_test = y.iloc[n_train:]

    ModelClass = get_model_class(model_type)
    model = ModelClass.load(f"models/model_{dataset}.json")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    r2 = r2_score(y_test, y_pred)

    os.makedirs("metrics", exist_ok=True)
    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4), "r2": round(r2, 4), "n_test": len(y_test)}
    with open(cfg["metrics_out"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas: {metrics}")

    # Gráfica: real vs predicho
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y.index, y.values, label="Real", color="steelblue")
    ax.plot(y_test.index, y_pred, label="Predicho (test)", color="darkorange", linewidth=2)
    ax.fill_between(
        y_test.index,
        y_pred - rmse,
        y_pred + rmse,
        alpha=0.2,
        color="darkorange",
        label="±1 RMSE",
    )
    ax.axvline(X_test.index[0], color="gray", linestyle="--", label="Inicio test")
    ax.set_title(f"Pronóstico vs Real — {dataset}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg["plot_out"], bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfica guardada en: {cfg['plot_out']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    evaluate(args.dataset)
