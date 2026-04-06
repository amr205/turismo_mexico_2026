"""
Archiva los resultados del experimento actual en carpetas con nombre del experimento.

Lee experiment.name de params.yaml y copia:
  - plots/forecast_*.png       → plots/{name}/
  - plots/shap_*.png           → plots/{name}/
  - plots/stl_*.png            → plots/{name}/
  - plots/forecast_future_*.png→ plots/{name}/
  - metrics/metrics_*.json     → metrics/{name}/

Uso:
    python src/archive_experiment.py
"""

import glob
import os
import shutil

import yaml


def archive() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    name = params.get("experiment", {}).get("name", "experimento")
    print(f"Archivando experimento: {name}")

    # Patrones de archivos a copiar
    plot_patterns = [
        "plots/forecast_*.png",
        "plots/shap_*.png",
        "plots/stl_*.png",
        "plots/forecast_future_*.png",
        "plots/backfill_comparison.png",
    ]
    metric_patterns = ["metrics/metrics_*.json"]

    dest_plots = f"plots/{name}"
    dest_metrics = f"metrics/{name}"
    os.makedirs(dest_plots, exist_ok=True)
    os.makedirs(dest_metrics, exist_ok=True)

    n_plots = 0
    for pattern in plot_patterns:
        for src in glob.glob(pattern):
            dst = os.path.join(dest_plots, os.path.basename(src))
            shutil.copy2(src, dst)
            n_plots += 1

    n_metrics = 0
    for pattern in metric_patterns:
        for src in glob.glob(pattern):
            dst = os.path.join(dest_metrics, os.path.basename(src))
            shutil.copy2(src, dst)
            n_metrics += 1

    print(f"  {n_plots} plots  → {dest_plots}/")
    print(f"  {n_metrics} métricas → {dest_metrics}/")
    print("Listo.")


if __name__ == "__main__":
    archive()
