"""
Entrenamiento del modelo de pronóstico de series de tiempo.

Uso:
    python src/train.py --dataset consumo
    python src/train.py --dataset ivf
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models import get_model

DATASET_CONFIG = {
    "consumo": {
        "features": "data/features/features_consumo.csv",
        "target_key": "consumo",
        "split_out": "models/split_consumo.json",
    },
    "ivf": {
        "features": "data/features/features_ivf.csv",
        "target_key": "ivf",
        "split_out": "models/split_ivf.json",
    },
}


def resolver_dataset(dataset: str, params: dict) -> tuple:
    """Devuelve (cfg, target_col) para datasets legacy o ivf_multi."""
    if dataset in DATASET_CONFIG:
        cfg = DATASET_CONFIG[dataset]
        target_col = params["target"][cfg["target_key"]]
        return cfg, target_col
    # Serie ivf_multi: deriva rutas por convención de nombres
    target_col = params["ivf_multi"][dataset]["target"]
    cfg = {
        "features": f"data/features/features_{dataset}.csv",
        "split_out": f"models/split_{dataset}.json",
    }
    return cfg, target_col


def train(dataset: str) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    cfg, target_col = resolver_dataset(dataset, params)
    model_type = params["model"]["type"]
    model_params = params["model"][model_type]
    test_size = params["train"]["test_size"]

    df = pd.read_csv(cfg["features"], parse_dates=["date"])
    df = df.set_index("date").sort_index()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # División cronológica (nunca aleatoria para series de tiempo)
    test_start = params["train"].get("test_start_date")
    if test_start:
        n_train = int((df.index < pd.Timestamp(test_start)).sum())
    else:
        n_train = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train = y.iloc[:n_train]

    model = get_model(model_type, model_params)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/model_{dataset}.json"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")

    split_meta = {
        "n_train": n_train,
        "n_test": len(X_test),
        "train_end_date": str(X_train.index[-1].date()),
        "test_start_date": str(X_test.index[0].date()),
        "model_type": model_type,
        "target_col": target_col,
    }
    with open(cfg["split_out"], "w") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Metadatos de división guardados en: {cfg['split_out']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    train(args.dataset)
