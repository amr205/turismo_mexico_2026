"""
Comparación de 5 métodos de backfill para indicadores de turismo pre-2018.

Lee los resultados de los experimentos DVC (dvc exp show --json) y genera:
  - plots/backfill_comparison.png  — figura 2×4: MAPE y R² por método y serie IVF
  - experiments/backfill_comparison.csv — tabla tidy con todas las métricas

Uso:
    python src/compare_backfill.py
    python src/compare_backfill.py --source /tmp/exps.json
    dvc exp show --json | python src/compare_backfill.py --source -
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["zero", "seasonal_mean", "seasonal_naive", "linear", "xgboost_backcast"]

METHOD_LABELS = {
    "zero": "Zero",
    "seasonal_mean": "Media estacional",
    "seasonal_naive": "Naïve estacional",
    "linear": "Regresión lineal",
    "xgboost_backcast": "XGBoost backcast",
}

SERIES = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios",
]

SERIES_LABELS = {
    "ivf_total_nacional": "IVF Total\nNacional",
    "ivf_turistico_total": "IVF Turístico\nTotal",
    "ivf_turistico_bienes": "IVF Turístico\nBienes",
    "ivf_turistico_servicios": "IVF Turístico\nServicios",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def cargar_experimentos(source: str | None) -> list:
    """
    Carga la salida de `dvc exp show --json`.

    source=None  → ejecuta el subproceso
    source="-"   → lee de stdin
    source=path  → lee el archivo JSON en path

    Devuelve la lista de objetos de experimento (puede estar vacía).
    """
    if source is None:
        dvc_bin = shutil.which("dvc")
        if dvc_bin is None:
            raise RuntimeError(
                "dvc no encontrado en PATH. Activa el entorno virtual:\n"
                "  source .venv/Scripts/activate"
            )
        result = subprocess.run(
            [dvc_bin, "exp", "show", "--json", "--no-pager"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"dvc exp show falló:\n{result.stderr}")
        raw = result.stdout.strip()
    elif source == "-":
        raw = sys.stdin.read().strip()
    else:
        with open(source, encoding="utf-8") as f:
            raw = f.read().strip()

    if not raw:
        raise RuntimeError(
            "La salida de dvc exp show está vacía. "
            "¿Corriste los 4 experimentos backcast_* primero?"
        )

    data = json.loads(raw)

    if not data:
        raise RuntimeError(
            "No se encontraron experimentos (lista vacía). "
            "Ejecuta primero:\n"
            "  dvc exp run --set-param features.indicators.backfill_method=zero "
            "--name backcast_zero\n  ... (y los otros 3 métodos)"
        )

    return data


# ---------------------------------------------------------------------------
# Extracción de registros
# ---------------------------------------------------------------------------


def _get_data(entrada: dict) -> dict:
    """
    Extrae el bloque 'data' de una entrada DVC, compatible con DVC 3.x.

    - Experimentos nombrados (dvc exp run --name):  entrada["revs"][0]["data"]
    - Baseline / workspace:                          entrada["data"]
    """
    revs = entrada.get("revs")
    if revs:
        return revs[0].get("data") or {}
    return entrada.get("data") or {}


def _metodo_de_entrada(entrada: dict) -> str | None:
    """Determina el método de backfill de una entrada de experimento."""
    name = entrada.get("name")
    if name and name.startswith("backcast_"):
        return name[len("backcast_"):]

    # Leer del bloque de params (estado base o workspace)
    data = _get_data(entrada)
    params_block = (data.get("params") or {}).get("params.yaml") or {}
    # DVC 3.x anida los valores bajo una clave "data"
    params_data = params_block.get("data") or params_block
    return (
        params_data.get("features", {})
        .get("indicators", {})
        .get("backfill_method")
    )


def _metricas_de_entrada(entrada: dict, serie: str) -> dict:
    """Extrae métricas de una serie concreta en una entrada de experimento."""
    data = _get_data(entrada)
    metrics = data.get("metrics") or {}

    # En Windows DVC usa backslash en las claves; probar ambos separadores
    key_fwd = f"metrics/metrics_{serie}.json"
    key_back = f"metrics\\metrics_{serie}.json"
    metricas_raw = metrics.get(key_fwd) or metrics.get(key_back) or {}

    # DVC 3.x anida los valores bajo una clave "data"
    if isinstance(metricas_raw, dict) and "data" in metricas_raw:
        metricas_raw = metricas_raw["data"]

    return {
        "mae":  metricas_raw.get("mae",  float("nan")),
        "rmse": metricas_raw.get("rmse", float("nan")),
        "mape": metricas_raw.get("mape", float("nan")),
        "r2":   metricas_raw.get("r2",   float("nan")),
    }


def _procesar_entrada(entrada: dict, registros: list) -> None:
    """Extrae registros de una entrada de experimento (baseline o experimento)."""
    metodo = _metodo_de_entrada(entrada)
    if metodo not in METHODS:
        return

    for serie in SERIES:
        m = _metricas_de_entrada(entrada, serie)
        registros.append({"method": metodo, "series": serie, **m})


def extraer_registros(exp_data: list) -> list:
    """
    Navega la estructura anidada de DVC 3.x y extrae registros planos.

    Maneja tanto la estructura anidada (DVC 3.x: baseline con 'experiments' list)
    como una lista plana (compatibilidad futura).
    """
    registros = []
    for baseline in exp_data:
        # El baseline mismo puede ser el workspace (xgboost_backcast comprometido)
        _procesar_entrada(baseline, registros)

        # Experimentos anidados (dvc exp run --name ...)
        for exp in baseline.get("experiments") or []:
            _procesar_entrada(exp, registros)

    return registros


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------


def validar_registros(registros: list) -> None:
    """Advierte si faltan combinaciones método×serie o si hay NaN."""
    df = pd.DataFrame(registros)
    if df.empty:
        print("ADVERTENCIA: no se extrajeron registros.")
        return

    combinaciones_encontradas = set(zip(df["method"], df["series"]))
    for m in METHODS:
        for s in SERIES:
            if (m, s) not in combinaciones_encontradas:
                print(f"ADVERTENCIA: falta combinación ({m}, {s})")

    for col in ["mape", "r2", "mae", "rmse"]:
        nulos = df[col].isna().sum()
        if nulos > 0:
            filas = df[df[col].isna()][["method", "series"]].to_string(index=False)
            print(f"ADVERTENCIA: {nulos} valor(es) NaN en '{col}':\n{filas}")

    # Detectar duplicados
    dupes = df.groupby(["method", "series"]).size()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        print(f"ADVERTENCIA: combinaciones duplicadas (se usará la primera):\n{dupes}")


# ---------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------


def construir_dataframe(registros: list) -> pd.DataFrame:
    """Devuelve DataFrame tidy con columnas: method, series, mape, r2, mae, rmse."""
    df = pd.DataFrame(registros)
    if df.empty:
        return df

    # Eliminar duplicados — mantener primera ocurrencia
    df = df.drop_duplicates(subset=["method", "series"], keep="first")

    # Ordenar por método en el orden canónico
    metodo_orden = {m: i for i, m in enumerate(METHODS)}
    df["_orden"] = df["method"].map(metodo_orden)
    df = df.sort_values(["_orden", "series"]).drop(columns=["_orden"])
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------


def graficar_comparacion(df: pd.DataFrame, out_path: str) -> None:
    """
    Figura 2 filas × 4 columnas:
      Fila 0: MAPE por serie (menor es mejor)
      Fila 1: R²   por serie (mayor es mejor)
    Cada columna corresponde a una serie IVF.
    Barras agrupadas por método, con paleta tab10 consistente.
    """
    colores = plt.colormaps["tab10"].colors
    color_por_metodo = {m: colores[i] for i, m in enumerate(METHODS)}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")

    handles_leyenda = []
    etiquetas_leyenda = []

    for col_idx, serie in enumerate(SERIES):
        df_s = df[df["series"] == serie].copy()
        df_s = df_s.set_index("method").reindex(METHODS)

        metodos_presentes = METHODS
        x = np.arange(len(metodos_presentes))
        ancho = 0.6

        for fila, (metrica, titulo_fila, mejor_fn) in enumerate([
            ("mape", "MAPE (%)", min),
            ("r2",   "R²",       max),
        ]):
            ax = axes[fila, col_idx]
            valores = df_s[metrica].values

            # Determinar barra óptima
            vals_validos = [v for v in valores if not (isinstance(v, float) and math.isnan(v))]
            mejor_val = mejor_fn(vals_validos) if vals_validos else None

            for i, (metodo, val) in enumerate(zip(metodos_presentes, valores)):
                color = color_por_metodo[metodo]
                es_mejor = (mejor_val is not None and not math.isnan(val) and val == mejor_val)
                borde_color = "black" if es_mejor else color
                borde_ancho = 2.5 if es_mejor else 0.8
                altura = val if not math.isnan(val) else 0

                barra = ax.bar(
                    x[i], altura,
                    width=ancho,
                    color=color,
                    edgecolor=borde_color,
                    linewidth=borde_ancho,
                    label=METHOD_LABELS[metodo] if (fila == 0 and col_idx == 0) else "_nolegend_",
                )

                if es_mejor:
                    ax.annotate(
                        "★",
                        xy=(x[i], altura),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=11,
                    )

                # Guardar para la leyenda
                if fila == 0 and col_idx == 0:
                    handles_leyenda.append(barra)
                    etiquetas_leyenda.append(METHOD_LABELS[metodo])

            # Línea de referencia R²=0
            if metrica == "r2":
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

            # Etiquetas de eje
            if col_idx == 0:
                ax.set_ylabel(titulo_fila, fontsize=10)

            if fila == 0:
                ax.set_title(SERIES_LABELS[serie], fontsize=10)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [METHOD_LABELS[m].replace(" ", "\n") for m in metodos_presentes],
                fontsize=7,
            )
            ax.tick_params(axis="x", rotation=0)

    fig.suptitle(
        "Comparación de métodos de backfill — precisión de pronóstico IVF\n"
        "(★ = mejor por panel)",
        fontsize=12,
    )

    # Leyenda compartida en la parte inferior
    if handles_leyenda:
        fig.legend(
            handles_leyenda,
            etiquetas_leyenda,
            loc="lower center",
            ncol=len(METHODS),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout(rect=[0, 0.07, 1, 0.97])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara 5 métodos de backfill usando resultados de experimentos DVC."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Fuente de datos: ruta a JSON, '-' para stdin, o None para ejecutar dvc.",
    )
    parser.add_argument(
        "--plot",
        default="plots/backfill_comparison.png",
        help="Ruta de salida para la figura comparativa.",
    )
    parser.add_argument(
        "--csv",
        default="experiments/backfill_comparison.csv",
        help="Ruta de salida para la tabla CSV.",
    )
    args = parser.parse_args()

    print("Cargando experimentos DVC...")
    exp_data = cargar_experimentos(args.source)

    print("Extrayendo registros...")
    registros = extraer_registros(exp_data)

    if not registros:
        print(
            "ERROR: no se encontraron registros de experimentos backcast_*.\n"
            "Verifica que los experimentos se hayan corrido correctamente."
        )
        sys.exit(1)

    validar_registros(registros)
    df = construir_dataframe(registros)

    print(f"\nMétodos encontrados: {sorted(df['method'].unique())}")
    print(f"Series encontradas:  {sorted(df['series'].unique())}")
    print(f"Total de registros:  {len(df)}\n")

    # Tabla de resumen en consola
    pivot_mape = df.pivot(index="method", columns="series", values="mape")
    pivot_r2   = df.pivot(index="method", columns="series", values="r2")
    print("MAPE (%) por método y serie:")
    print(pivot_mape.round(3).to_string())
    print("\nR² por método y serie:")
    print(pivot_r2.round(4).to_string())

    # Guardar CSV
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"\nTabla CSV guardada en: {args.csv}")

    # Generar figura
    graficar_comparacion(df, args.plot)


if __name__ == "__main__":
    main()
