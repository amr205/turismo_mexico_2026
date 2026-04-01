# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python DVC pipeline for academic research on Mexican tourism economics (INEGI data). Covers data cleaning, feature engineering, time series forecasting (XGBoost), evaluation, and interpretation (SHAP + STL decomposition).

## Setup

```bash
pip install -r requirements.txt
```

Always run DVC commands with the virtual environment active:
```bash
source .venv/Scripts/activate   # Windows (bash)
```

## Running the Pipeline

```bash
# Run the full pipeline (only re-runs changed stages)
dvc repro

# Force re-run a specific stage and its dependents
dvc repro train_consumo

# Force re-run everything
dvc repro --force
```

## Experiment Tracking

```bash
# Run an experiment with different hyperparameters
dvc exp run --set-param model.xgboost.max_depth=5 --name "depth5"
dvc exp run --set-param features.lags="[1,2,4,8]" --name "lags_extended"
dvc exp run --set-param model.type=lstm --name "lstm_model"   # once lstm_model.py is implemented

# Compare experiments
dvc exp show
dvc exp show --only-changed
dvc exp show --csv > experiments_summary.csv

# Apply best experiment and commit
dvc exp apply <experiment-name>
git add params.yaml dvc.lock metrics/ plots/
git commit -m "feat: adopt <experiment-name>"

# View current metrics
dvc metrics show
```

## Architecture

**DVC Pipeline (dvc.yaml) — 11 stages:**

```
data/raw/
  ├── clean_consumo    → data/processed/consumo_turistico_inegi_clean.csv
  ├── clean_ivf        → data/processed/indice_volumen_fisico_inegi_clean.csv
  ├── clean_indicadores→ data/processed/turismo_indicadores_inegi_clean.csv
  └── plot_timeseries  → plots/consumo_turistico.png, plots/ivf_turismo.png

data/processed/  (quarterly, ~131 rows each)
  ├── features_consumo → data/features/features_consumo.csv (lag + rolling + dummies)
  └── features_ivf     → data/features/features_ivf.csv

data/features/
  ├── train_consumo    → models/xgboost_consumo.json, models/split_consumo.json
  └── train_ivf        → models/xgboost_ivf.json, models/split_ivf.json

models/
  ├── evaluate_consumo → metrics/metrics_consumo.json, plots/forecast_consumo.png
  ├── evaluate_ivf     → metrics/metrics_ivf.json, plots/forecast_ivf.png
  ├── interpret_consumo→ plots/shap_consumo.png, plots/stl_consumo.png
  └── interpret_ivf    → plots/shap_ivf.png, plots/stl_ivf.png
```

**Pipeline scripts** use `--dataset {consumo,ivf}` to serve both datasets:
- [src/features.py](src/features.py) — lag features, rolling means, quarter dummies, trend
- [src/train.py](src/train.py) — chronological train/test split, fits model from registry
- [src/evaluate.py](src/evaluate.py) — MAE, RMSE, MAPE, R², forecast plot
- [src/interpret.py](src/interpret.py) — SHAP bar chart + STL decomposition

**All tunable values live in [params.yaml](params.yaml)** — changing any value triggers DVC to re-run affected stages.

## Model Registry (`src/models/`)

Adding a new model type requires three steps only, without touching pipeline scripts:

1. Create `src/models/my_model.py` implementing `BaseForecaster` ([src/models/base.py](src/models/base.py))
2. Register it in [src/models/__init__.py](src/models/__init__.py)
3. Add its hyperparameter block under `model:` in [params.yaml](params.yaml)
4. Set `model.type: my_model` and run `dvc repro`

`BaseForecaster` requires: `fit`, `predict`, `save`, `load`, `get_shap_explainer`.
- XGBoost uses `shap.TreeExplainer` (fast, exact)
- Future neural nets should use `shap.DeepExplainer` or `shap.KernelExplainer`

## Data Sources

All raw data from INEGI (Instituto Nacional de Estadística y Geografía):
- `consumo_turistico`: quarterly tourism consumption 1993–present
- `indice_volumen_fisico`: quarterly physical volume index 1993–present
- `turismo_indicadores`: monthly visitor flows 2018–present (pivoted to long format, ~320K rows)

## Conventions

- Comments and variable names are in Spanish (domain terminology).
- Scripts run from the project root (all paths are root-relative).
- Chronological split only in `train.py` — never random shuffle for time series.
- `numpy<2.0` is pinned in requirements.txt for Python 3.10.4 compatibility across xgboost, shap, and statsmodels.
