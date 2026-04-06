"""
Pronóstico recursivo multi-paso para las series IVF.

Genera predicciones desde el último dato observado hasta el horizonte definido en
params.yaml (forecast.horizon trimestres). Para trimestres futuros sin datos reales
de indicadores (post 2025-Q4) aplica carry-forward del mismo trimestre del año anterior.

Uso:
    python src/forecast.py --series ivf_total_nacional
    python src/forecast.py --series ivf_turistico_total --horizon 8
"""

import argparse
import json
import os
import sys
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models import get_model_class

SERIES_VALIDAS = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]

IVF_PATH = "data/processed/indice_volumen_fisico_inegi_clean.csv"
BACKCASTED_PATH = "data/processed/indicadores_backcasted.csv"


def cargar_indicadores_trimestrales() -> pd.DataFrame:
    """Carga el archivo de indicadores con backfill ya aplicado."""
    df = pd.read_csv(BACKCASTED_PATH, parse_dates=["date"])
    return df.set_index("date").sort_index()


def extender_indicadores(df_q: pd.DataFrame, ultima_fecha: pd.Timestamp, horizon: int, metodo: str) -> pd.DataFrame:
    """
    Extiende el DataFrame de indicadores trimestrales hasta cubrir el horizonte de pronóstico.
    Método 'same_quarter_prior_year': rellena cada trimestre faltante con el valor del
    mismo trimestre del año anterior.
    """
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=3),
        periods=horizon,
        freq="QS",
    )
    filas_nuevas = []
    for fecha in fechas_futuras:
        if fecha in df_q.index:
            continue
        # Buscar mismo trimestre del año anterior
        fecha_ref = fecha - pd.DateOffset(years=1)
        if fecha_ref in df_q.index:
            fila = df_q.loc[fecha_ref].copy()
        else:
            fila = pd.Series(0.0, index=df_q.columns)
        fila.name = fecha
        filas_nuevas.append(fila)

    if filas_nuevas:
        df_ext = pd.concat([df_q, pd.DataFrame(filas_nuevas)])
    else:
        df_ext = df_q.copy()
    return df_ext.sort_index()


def forecast(series: str, horizon: int) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    target_col = params["ivf_multi"][series]["target"]
    feat_params = params["features"]
    lags = feat_params["lags"]
    windows = feat_params["rolling_windows"]
    metodo_ind = params["forecast"].get("future_indicator_method", "same_quarter_prior_year")

    model_type = params["model"]["type"]
    features_path = f"data/features/features_{series}.csv"
    model_path = f"models/model_{series}.json"
    split_path = f"models/split_{series}.json"

    # --- Cargar datos entrenados ---
    df_feat = pd.read_csv(features_path, parse_dates=["date"])
    df_feat = df_feat.set_index("date").sort_index()

    with open(split_path) as f:
        split = json.load(f)
    n_train = split["n_train"]
    rmse_approx = None  # Se calcula abajo con predicciones test

    ModelClass = get_model_class(model_type)
    model = ModelClass.load(model_path)

    # Calcular RMSE en test para bandas de incertidumbre
    X_full = df_feat.drop(columns=[target_col])
    y_full = df_feat[target_col]
    X_test = X_full.iloc[n_train:]
    y_test = y_full.iloc[n_train:]
    y_pred_test = model.predict(X_test)
    rmse_approx = float(np.sqrt(np.mean((y_test.values - y_pred_test) ** 2)))

    # --- Buffer de historia (valores reales observados) ---
    df_ivf = pd.read_csv(IVF_PATH, parse_dates=["date"])
    df_ivf = df_ivf.set_index("date").sort_index()
    historia = list(df_ivf[target_col].dropna().values)
    fechas_hist = list(df_ivf[target_col].dropna().index)

    ultima_fecha = fechas_hist[-1]
    n_total_obs = len(df_feat)  # filas en features (post-dropna, desde lag máximo)
    trend_ultimo = int(df_feat["trend"].iloc[-1])

    # Identificar columnas de indicadores
    base_cols = {target_col} | {f"lag_{l}" for l in lags} | {f"roll_mean_{w}" for w in windows}
    base_cols |= {f"q{q}" for q in [2, 3, 4]} | {"trend"}
    ind_cols = [c for c in df_feat.columns if c not in base_cols]

    # --- Cargar y extender indicadores ---
    df_ind_q = cargar_indicadores_trimestrales()
    df_ind_q = extender_indicadores(df_ind_q, ultima_fecha, horizon, metodo_ind)

    # --- Pronóstico recursivo ---
    predicciones = []
    fechas_pred = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=3),
        periods=horizon,
        freq="QS",
    )

    historia_buffer = historia.copy()  # crece con cada predicción

    for step, fecha_pred in enumerate(fechas_pred):
        # Features autorregresivas
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = historia_buffer[-(lag)]
        for window in windows:
            # roll_mean_w en t = mean(t-1, ..., t-w) por el shift(1) en construcción
            row[f"roll_mean_{window}"] = float(np.mean(historia_buffer[-(window + 1):-1]))
        # Dummies de trimestre
        q = fecha_pred.quarter
        for qd in [2, 3, 4]:
            row[f"q{qd}"] = int(q == qd)
        # Tendencia lineal extrapolada
        row["trend"] = trend_ultimo + step + 1

        # Indicadores: valor del trimestre correspondiente (o cero si no existe)
        if fecha_pred in df_ind_q.index:
            ind_vals = df_ind_q.loc[fecha_pred]
        else:
            ind_vals = pd.Series(0.0, index=ind_cols)
        for col in ind_cols:
            row[col] = float(ind_vals.get(col, 0.0))

        # Predicción
        X_row = pd.DataFrame([row])[list(X_full.columns)]
        y_hat = float(model.predict(X_row)[0])

        predicciones.append({"date": fecha_pred, "predicted": y_hat})
        historia_buffer.append(y_hat)

    # --- Guardar CSV ---
    os.makedirs("data/forecasts", exist_ok=True)
    df_pred = pd.DataFrame(predicciones)
    df_pred["lower_1rmse"] = df_pred["predicted"] - rmse_approx
    df_pred["upper_1rmse"] = df_pred["predicted"] + rmse_approx
    df_pred["series"] = series
    out_csv = f"data/forecasts/forecast_{series}.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"Pronóstico guardado en: {out_csv}")

    # --- Gráfica ---
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    # Historia completa
    ax.plot(fechas_hist, historia, color="steelblue", label="Observado", linewidth=1.2)

    # Predicciones en test (para contexto)
    fechas_test = df_feat.index[n_train:]
    ax.plot(fechas_test, y_pred_test, color="darkorange", linestyle="--",
            linewidth=1.5, label="Ajuste test")

    # Pronóstico futuro
    ax.plot(df_pred["date"], df_pred["predicted"], color="crimson",
            linewidth=2, marker="o", markersize=4, label="Pronóstico")
    ax.fill_between(
        df_pred["date"],
        df_pred["lower_1rmse"],
        df_pred["upper_1rmse"],
        color="crimson",
        alpha=0.15,
        label="±1 RMSE",
    )
    ax.axvline(ultima_fecha, color="gray", linestyle=":", linewidth=1, label="Último dato")
    ax.set_title(f"Pronóstico {series} — horizonte {horizon} trimestres")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(target_col)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out_plot = f"plots/forecast_future_{series}.png"
    fig.savefig(out_plot, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en: {out_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=SERIES_VALIDAS, required=True)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Trimestres a pronosticar (default: forecast.horizon en params.yaml)")
    args = parser.parse_args()

    with open("params.yaml") as f:
        _params = yaml.safe_load(f)
    horizon = args.horizon or _params["forecast"]["horizon"]

    forecast(args.series, horizon)
