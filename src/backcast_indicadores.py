"""
Backfill de indicadores de turismo para trimestres anteriores a 2018.

Genera estimaciones de los 39 indicadores trimestrales para el período 1993-2018
usando distintos métodos, y produce un archivo ancho listo para usarse en features.

Métodos disponibles (features.indicators.backfill_method en params.yaml):
  zero            → rellena con 0 (baseline — crea ruptura estructural artificial)
  seasonal_mean   → media del mismo trimestre sobre 2018-2025
  seasonal_naive  → lleva hacia atrás el valor del mismo trimestre del año más próximo
  linear          → regresión lineal por indicador, extrapola tendencia hacia atrás
  xgboost_backcast→ invierte la serie, entrena XGBoost, predice hacia atrás

Salida:
  data/processed/indicadores_backcasted.csv
    Formato ancho: filas = trimestres (QS), columnas = combinaciones de indicadores.
    Cubre desde el primer trimestre IVF hasta el último dato de indicadores.

Uso:
    python src/backcast_indicadores.py
"""

import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

IVF_PATH = "data/processed/indice_volumen_fisico_inegi_clean.csv"
IND_PATH = "data/processed/turismo_indicadores_inegi_clean.csv"
OUT_PATH = "data/processed/indicadores_backcasted.csv"


# ── Utilidades ────────────────────────────────────────────────────────────────

def normalizar_col(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    return s.replace(" ", "_").replace(".", "").replace(",", "").replace("-", "_").lower()


def cargar_indicadores_trimestrales() -> pd.DataFrame:
    """Carga turismo_indicadores y agrega a trimestral (suma)."""
    df = pd.read_csv(IND_PATH, parse_dates=["fecha"])
    df = df.dropna(subset=["variable", "tipo", "movilidad", "flujo", "valor"])
    df["combo"] = (
        df["variable"].apply(normalizar_col)
        + "__"
        + df["tipo"].apply(normalizar_col)
        + "__"
        + df["movilidad"].apply(normalizar_col)
        + "__"
        + df["flujo"].apply(normalizar_col)
    )
    pivot = df.pivot_table(index="fecha", columns="combo", values="valor", aggfunc="sum")
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.resample("QS").sum()


def rango_completo(df_ivf: pd.DataFrame, df_q: pd.DataFrame) -> pd.DatetimeIndex:
    """Índice trimestral completo desde el primer dato IVF hasta el último indicador."""
    inicio = df_ivf.index.min()
    fin = df_q.index.max()
    return pd.date_range(start=inicio, end=fin, freq="QS")


# ── Métodos de backfill ───────────────────────────────────────────────────────

def metodo_zero(df_q: pd.DataFrame, idx_completo: pd.DatetimeIndex) -> pd.DataFrame:
    """Rellena con cero todos los trimestres anteriores al primer dato real."""
    df_full = df_q.reindex(idx_completo)
    return df_full.fillna(0.0)


def metodo_seasonal_mean(df_q: pd.DataFrame, idx_completo: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Media de cada trimestre (Q1..Q4) calculada sobre los datos disponibles (2018-2025).
    Pre-2018: cada Q1 → media de todos los Q1 disponibles, etc.
    """
    df_full = df_q.reindex(idx_completo)
    mask_pre = df_full.index < df_q.index.min()

    # Media por trimestre
    df_q_tmp = df_q.copy()
    df_q_tmp["_q"] = df_q_tmp.index.quarter
    medias = df_q_tmp.groupby("_q").mean().drop(columns=["_q"], errors="ignore")

    for fecha in df_full.index[mask_pre]:
        q = fecha.quarter
        df_full.loc[fecha] = medias.loc[q]

    return df_full.fillna(0.0)


def metodo_seasonal_naive(df_q: pd.DataFrame, idx_completo: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Para cada trimestre pre-2018, usa el valor del mismo trimestre del año inmediatamente
    siguiente con datos reales (carry-backward estacional).
    """
    df_full = df_q.reindex(idx_completo)
    pre = df_full.index[df_full.index < df_q.index.min()]

    for fecha in sorted(pre, reverse=True):  # de más reciente a más antiguo
        for lag_anios in range(1, 30):
            ref = fecha + pd.DateOffset(years=lag_anios)
            if ref in df_q.index:
                df_full.loc[fecha] = df_q.loc[ref]
                break

    return df_full.fillna(0.0)


def metodo_linear(df_q: pd.DataFrame, idx_completo: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Regresión lineal (tendencia + dummies de trimestre) por cada columna de indicador.
    Se ajusta sobre 2018-2025 y se extrapola hacia atrás. Valores negativos → 0.
    """
    df_full = df_q.reindex(idx_completo).copy()
    idx_all = idx_completo
    trend_all = np.arange(len(idx_all))
    q_all = idx_all.quarter

    # Dummies de trimestre
    Q = np.column_stack([
        trend_all,
        (q_all == 2).astype(int),
        (q_all == 3).astype(int),
        (q_all == 4).astype(int),
    ])

    # Máscara de filas con datos reales
    mask_real = idx_all >= df_q.index.min()
    X_train = Q[mask_real]
    X_pred = Q[~mask_real]

    for col in df_full.columns:
        y_train = df_q[col].reindex(idx_all[mask_real]).values
        if np.all(y_train == 0):
            df_full.loc[idx_all[~mask_real], col] = 0.0
            continue
        reg = LinearRegression().fit(X_train, y_train)
        y_back = reg.predict(X_pred)
        y_back = np.clip(y_back, 0, None)  # indicadores no pueden ser negativos
        df_full.loc[idx_all[~mask_real], col] = y_back

    return df_full.fillna(0.0)


def _build_features_ts(
    serie: pd.Series, lags: list, windows: list, quarter_orig: pd.Index
) -> pd.DataFrame:
    """
    Construye matriz de features autorregresiva para una serie temporal.
    quarter_orig: índice de trimestres originales (en orden real, no invertido) para dummies.
    """
    df = pd.DataFrame(index=serie.index)
    for lag in lags:
        df[f"lag_{lag}"] = serie.shift(lag)
    for w in windows:
        df[f"roll_{w}"] = serie.shift(1).rolling(w).mean()
    # quarter_orig está en orden real; para la serie invertida, los trimestres van al revés
    q_inv = quarter_orig[::-1].values  # invertir igual que la serie
    for qd in [2, 3, 4]:
        df[f"q{qd}"] = (q_inv == qd).astype(int)
    df["trend"] = np.arange(len(serie))
    return df.dropna()


def metodo_xgboost_backcast(
    df_q: pd.DataFrame, idx_completo: pd.DatetimeIndex, params: dict
) -> pd.DataFrame:
    """
    Para cada indicador:
      1. Invierte la serie (2025-Q4 → 2018-Q3) para que el pasado sea el "futuro".
      2. Entrena XGBoost sobre la serie invertida.
      3. Predice recursivamente hacia "adelante" en tiempo invertido = hacia atrás en real.
      4. Invierte el resultado para obtener estimaciones 1993-2017.
    Con max_depth pequeño y n_estimators reducido para evitar sobreajuste con ~26 puntos.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("xgboost no disponible, usando seasonal_mean como fallback.")
        return metodo_seasonal_mean(df_q, idx_completo)

    lags = [1, 2, 3, 4]
    windows = [2, 4]
    n_lag_max = max(lags)
    n_roll_max = max(windows)
    min_obs = n_lag_max + n_roll_max + 1  # mínimo para evitar dropna total

    xgb_params = {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "verbosity": 0,
    }

    df_full = df_q.reindex(idx_completo).copy()
    pre_idx = idx_completo[idx_completo < df_q.index.min()]
    n_back = len(pre_idx)

    if n_back == 0:
        return df_full.fillna(0.0)

    cols_ok = 0
    cols_fallback = 0

    for col in df_q.columns:
        serie = df_q[col].copy()

        if serie.isna().all() or (serie == 0).all() or len(serie) < min_obs:
            df_full.loc[pre_idx, col] = 0.0
            cols_fallback += 1
            continue

        # 1. Invertir la serie en el tiempo
        serie_inv = serie.iloc[::-1].reset_index(drop=True)
        serie_inv.index = pd.RangeIndex(len(serie_inv))

        # 2. Construir features sobre la serie invertida
        feat = _build_features_ts(serie_inv, lags, windows, serie.index)
        if len(feat) < 5:
            df_full.loc[pre_idx, col] = 0.0
            cols_fallback += 1
            continue

        X = feat.drop(columns=[])  # todas las columnas son features
        # La "y" es la serie invertida desplazada 0 (el valor a predecir en la posición invertida)
        # En realidad usamos la serie invertida misma como y, alineada con feat
        y = serie_inv.loc[feat.index]

        model = XGBRegressor(**xgb_params)
        model.fit(X, y)

        # 3. Predicción recursiva hacia atrás
        # El buffer empieza con los valores REALES de la serie original en orden invertido
        buffer = list(serie.values[::-1])  # [2025-Q4, 2025-Q3, ..., 2018-Q3]

        predicciones_inv = []
        # Los trimestres pre-2018 en orden inverso (2018-Q2 → 1993-Q1)
        fechas_back_inv = pre_idx[::-1]

        for step, fecha_real in enumerate(fechas_back_inv):
            row = {}
            for lag in lags:
                row[f"lag_{lag}"] = buffer[lag - 1] if lag - 1 < len(buffer) else 0.0
            for w in windows:
                vals = buffer[1 : w + 1]  # shift(1) + rolling
                row[f"roll_{w}"] = float(np.mean(vals)) if vals else 0.0
            q_real = fecha_real.quarter
            for qd in [2, 3, 4]:
                row[f"q{qd}"] = int(q_real == qd)
            # El trend en la serie invertida sigue creciendo
            row["trend"] = len(buffer)

            X_row = pd.DataFrame([row])[X.columns]
            y_hat = float(model.predict(X_row)[0])
            y_hat = max(y_hat, 0.0)  # no negativos

            predicciones_inv.append(y_hat)
            buffer.insert(0, y_hat)  # prepend al buffer invertido

        # 4. Invertir predicciones → orden cronológico (1993-Q1 → 2018-Q2)
        predicciones = list(reversed(predicciones_inv))
        df_full.loc[pre_idx, col] = predicciones
        cols_ok += 1

    print(
        f"  xgboost_backcast: {cols_ok} indicadores backcasted, "
        f"{cols_fallback} con fallback a 0"
    )
    return df_full.fillna(0.0)


# ── Dispatcher ────────────────────────────────────────────────────────────────

METODOS = {
    "zero": metodo_zero,
    "seasonal_mean": metodo_seasonal_mean,
    "seasonal_naive": metodo_seasonal_naive,
    "linear": metodo_linear,
    "xgboost_backcast": metodo_xgboost_backcast,
}


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    backfill_method = (
        params.get("features", {})
        .get("indicators", {})
        .get("backfill_method", "zero")
    ) or "zero"

    if backfill_method not in METODOS:
        raise ValueError(
            f"backfill_method '{backfill_method}' no reconocido. "
            f"Opciones: {list(METODOS.keys())}"
        )

    print(f"Método de backfill: {backfill_method}")

    # Cargar datos
    df_ivf = pd.read_csv(IVF_PATH, parse_dates=["date"]).set_index("date").sort_index()
    df_q = cargar_indicadores_trimestrales()
    idx_completo = rango_completo(df_ivf, df_q)

    n_pre = (idx_completo < df_q.index.min()).sum()
    print(
        f"  Trimestres totales: {len(idx_completo)} | "
        f"Con datos reales: {len(df_q)} | "
        f"A backcastear: {n_pre}"
    )

    # Ejecutar método
    fn = METODOS[backfill_method]
    if backfill_method == "xgboost_backcast":
        df_out = fn(df_q, idx_completo, params)
    else:
        df_out = fn(df_q, idx_completo)

    # Guardar
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.index.name = "date"
    df_out.reset_index().to_csv(OUT_PATH, index=False)
    print(
        f"Indicadores backcasted guardados en: {OUT_PATH} "
        f"({df_out.shape[0]} filas × {df_out.shape[1]} columnas)"
    )

    # Estadísticas de cobertura
    n_nonzero = (df_out.loc[idx_completo[idx_completo < df_q.index.min()]] != 0).any(axis=1).sum()
    print(f"  Trimestres pre-2018 con al menos un valor != 0: {n_nonzero} / {n_pre}")


if __name__ == "__main__":
    main()
