# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python DVC pipeline for academic research on Mexican tourism economics (INEGI data). Covers data cleaning, feature engineering, time series forecasting (XGBoost + 4 NN architectures), evaluation, and interpretation (SHAP + STL decomposition).

**Branch `paper_experiments`**: ablation study comparing 5 model types × 3 backfill methods = 15 experiments. Test set is post-COVID (2022 Q1–2025 Q3). COVID dummy covers 2020 Q1–2021 Q4.

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
dvc repro train_ivf_multi@ivf_total_nacional

# Force re-run everything
dvc repro --force
```

## Experiment Tracking

```bash
# Run a single experiment manually
dvc exp run --name xgb_linear \
  --set-param experiment.name=xgb_linear \
  --set-param model.type=xgboost \
  --set-param features.indicators.backfill_method=linear

# Apply experiment to working directory (restores params + outputs)
dvc exp apply xgb_linear

# Archive to plots/{name}/ and metrics/{name}/
python src/archive_experiment.py

# Compare all experiments
dvc exp show
dvc exp show --only-changed
dvc exp show --csv > experiments_summary.csv

# Run all 15 paper experiments automatically
python src/run_all_experiments.py
python src/run_all_experiments.py --dry-run
python src/run_all_experiments.py --only xgb_linear,mlp_linear
```

## Architecture

**DVC Pipeline (dvc.yaml) — 20+ stages:**

```
data/raw/
  ├── clean_consumo / clean_ivf / clean_indicadores
  └── plot_timeseries / plot_ivf_overview

data/processed/
  ├── features_consumo / features_ivf    (legacy series)
  └── backcast_indicadores               (fills pre-2018 gaps in turismo_indicadores)

backcast + ivf cleaned →
  features_ivf_multi[ivf_total_nacional | ivf_turistico_total | ivf_turistico_bienes | ivf_turistico_servicios]

features_ivf_multi → train_ivf_multi → evaluate_ivf_multi → interpret_ivf_multi → forecast_ivf_multi
                                                                ↓
                                                         compare_shap
```

**Key params.yaml sections:**
- `features.covid_dummy: true` — adds binary dummy 1 for 2020 Q1–2021 Q4
- `train.test_start_date: "2022-01-01"` — overrides test_size, uses date-based split
- `features.indicators.backfill_method` — `zero | linear | xgboost_backcast | seasonal_mean | seasonal_naive`
- `experiment.name` — used by `archive_experiment.py` to name output folders
- `model.type` — `xgboost | mlp | gru | cnn_gru | res_cnn_gru`

## Model Registry (`src/models/`)

5 model types are registered. Adding a new model:

1. Create `src/models/my_model.py` implementing `BaseForecaster` ([src/models/base.py](src/models/base.py))
2. Register it in [src/models/__init__.py](src/models/__init__.py)
3. Add its hyperparameter block under `model:` in [params.yaml](params.yaml)
4. Set `model.type: my_model` and run `dvc repro`

`BaseForecaster` requires: `fit`, `predict`, `save`, `load`, `get_shap_explainer`.
- XGBoost: `shap.TreeExplainer`
- MLP: `shap.GradientExplainer` directly on `(n, n_features)` input
- GRU / CNN-GRU / Res-CNN-GRU: `shap.GradientExplainer`, SHAP values averaged over sequence dim

**Sequence models** (GRU, CNN-GRU, Res-CNN-GRU): store `context_X = X_train[-(seq_len-1):]` during `fit()`, prepend it in `predict()` to reconstruct full sequences at test time without data leakage.

**Save format** (PyTorch models): `torch.save({"state_dict": ..., "init_params": ..., "context_X": ...}, path)`

## Data Sources

All raw data from INEGI (Instituto Nacional de Estadística y Geografía):
- `consumo_turistico`: quarterly tourism consumption 1993–present
- `indice_volumen_fisico`: quarterly physical volume index 1993–present (4 series: total nacional, turístico total/bienes/servicios)
- `turismo_indicadores`: monthly visitor flows 2018–present (pivoted to long format, ~320K rows)

## Conventions

- Comments and variable names are in Spanish (domain terminology).
- Scripts run from the project root (all paths are root-relative).
- Chronological split only — never random shuffle for time series.
- `numpy<2.0` is pinned in requirements.txt for Python 3.10.4 compatibility across xgboost, shap, and statsmodels.
- `dvc exp run` (not `dvc repro`) is the standard way to run experiments so they are tracked and comparable via `dvc exp show`.
