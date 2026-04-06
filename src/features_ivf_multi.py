"""
Construcción de matriz de características para forecast multiserie IVF con indicadores externos.

Para cada una de las 4 series IVF combina:
  - Features autorregresivas: rezagos, medias móviles, dummies de trimestre, tendencia.
  - Indicadores de turismo INEGI: num_visitantes, gasto_total, gasto_medio agregados a
    frecuencia trimestral (suma). Pre-2018 se rellena con ceros.

Uso:
    python src/features_ivf_multi.py --series ivf_total_nacional
    python src/features_ivf_multi.py --series ivf_turistico_total
    python src/features_ivf_multi.py --series ivf_turistico_bienes
    python src/features_ivf_multi.py --series ivf_turistico_servicios
"""

import argparse
import unicodedata

import pandas as pd
import yaml

SERIES_VALIDAS = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]

IVF_PATH = "data/processed/indice_volumen_fisico_inegi_clean.csv"
IND_PATH = "data/processed/turismo_indicadores_inegi_clean.csv"
BACKCASTED_PATH = "data/processed/indicadores_backcasted.csv"


def normalizar_col(s: str) -> str:
    """Convierte nombre a identificador ASCII seguro."""
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    return s.replace(" ", "_").replace(".", "").replace(",", "").replace("-", "_").lower()


def build_features(series: str) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    target_col = params["ivf_multi"][series]["target"]
    feat_params = params["features"]
    fill_val = feat_params.get("indicators", {}).get("fill_missing", 0)

    # --- 1. Cargar IVF y construir features autorregresivas ---
    df = pd.read_csv(IVF_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    serie = df[target_col].copy()

    features = pd.DataFrame(index=serie.index)
    features[target_col] = serie

    # Rezagos
    for lag in feat_params["lags"]:
        features[f"lag_{lag}"] = serie.shift(lag)

    # Medias móviles (shift(1) para evitar fuga de datos)
    for window in feat_params["rolling_windows"]:
        features[f"roll_mean_{window}"] = serie.shift(1).rolling(window).mean()

    # Dummies de trimestre (q1 es categoría de referencia, se omite)
    if feat_params["add_quarter_dummies"]:
        quarter = serie.index.quarter
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

    # Eliminar NaN por rezagos/rolling
    features = features.dropna()

    # Recorte temporal: si train_start_date está definido, se descarta todo lo anterior
    train_start = feat_params.get("train_start_date")
    if train_start:
        features = features[features.index >= pd.Timestamp(train_start)]
        print(f"  Recorte temporal activo: desde {train_start} ({len(features)} trimestres)")

    # --- 2. Cargar indicadores pre-procesados (con backfill aplicado por backcast_indicadores) ---
    df_ind = pd.read_csv(BACKCASTED_PATH, parse_dates=["date"])
    df_ind = df_ind.set_index("date").sort_index()

    # --- 3. Merge izquierdo: conserva todos los trimestres IVF ---
    features = features.merge(df_ind, left_index=True, right_index=True, how="left")

    # Rellenar NaN residuales (combinaciones faltantes por mezcla de índices)
    ind_cols = list(df_ind.columns)
    features[ind_cols] = features[ind_cols].fillna(fill_val)

    # --- 4. Guardar ---
    out_path = f"data/features/features_{series}.csv"
    features.reset_index().to_csv(out_path, index=False)
    print(
        f"Características guardadas en: {out_path} "
        f"({len(features)} filas, {features.shape[1]} columnas)"
    )
    print(f"  Features autorregresivas: {features.shape[1] - len(ind_cols) - 1}")
    print(f"  Columnas de indicadores:  {len(ind_cols)}")
    n_con_ind = (features[ind_cols].sum(axis=1) != 0).sum()
    print(f"  Trimestres con indicadores reales: {n_con_ind} / {len(features)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=SERIES_VALIDAS, required=True)
    args = parser.parse_args()
    build_features(args.series)
