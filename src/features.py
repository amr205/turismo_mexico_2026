"""
Construcción de matriz de características para modelos de pronóstico.

Uso:
    python src/features.py --dataset consumo
    python src/features.py --dataset ivf
"""

import argparse
import pandas as pd
import yaml

DATASET_CONFIG = {
    "consumo": {
        "input": "data/processed/consumo_turistico_inegi_clean.csv",
        "output": "data/features/features_consumo.csv",
        "target_key": "consumo",
    },
    "ivf": {
        "input": "data/processed/indice_volumen_fisico_inegi_clean.csv",
        "output": "data/features/features_ivf.csv",
        "target_key": "ivf",
    },
}


def build_features(dataset: str) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    cfg = DATASET_CONFIG[dataset]
    target_col = params["target"][cfg["target_key"]]
    feat_params = params["features"]

    df = pd.read_csv(cfg["input"], parse_dates=["date"])
    df = df.set_index("date").sort_index()

    series = df[target_col].copy()

    features = pd.DataFrame(index=series.index)
    features[target_col] = series

    # Características de rezago
    for lag in feat_params["lags"]:
        features[f"lag_{lag}"] = series.shift(lag)

    # Medias móviles (shift(1) antes de rolling para evitar fuga de datos)
    for window in feat_params["rolling_windows"]:
        features[f"roll_mean_{window}"] = series.shift(1).rolling(window).mean()

    # Dummies de trimestre (se omite q1 como referencia)
    if feat_params["add_quarter_dummies"]:
        quarter = series.index.quarter
        for q in [2, 3, 4]:
            features[f"q{q}"] = (quarter == q).astype(int)

    # Dummy COVID: ruptura estructural 2020 Q1 – 2021 Q4
    if feat_params.get("covid_dummy", False):
        features["covid"] = (
            (features.index >= pd.Timestamp("2020-01-01")) &
            (features.index <= pd.Timestamp("2021-10-01"))
        ).astype(int)

    # Tendencia lineal
    if feat_params["add_trend"]:
        features["trend"] = range(len(features))

    # Eliminar filas con NaN generadas por rezagos/rolling
    features = features.dropna()

    features.reset_index().to_csv(cfg["output"], index=False)
    print(f"Características guardadas en: {cfg['output']} ({len(features)} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["consumo", "ivf"], required=True)
    args = parser.parse_args()
    build_features(args.dataset)
