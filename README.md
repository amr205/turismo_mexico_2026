# Análisis de Turismo en México

Repositorio de código para el artículo académico sobre economía del turismo en México. Implementa un pipeline reproducible con DVC para limpiar datos del INEGI, entrenar modelos de pronóstico de series de tiempo e interpretar los resultados.

---

## Descripción del proyecto

El proyecto procesa tres fuentes de datos trimestrales/mensuales del **INEGI** (Instituto Nacional de Estadística y Geografía):

| Dataset | Fuente | Frecuencia | Período |
|---|---|---|---|
| Consumo turístico | `data/raw/consumo_turistico_inegi.csv` | Trimestral | 1993–presente |
| Índice de volumen físico (IVF) | `data/raw/indice_volumen_fisico_inegi.csv` | Trimestral | 1993–presente |
| Indicadores de visitantes | `data/raw/turismo_indicadores_inegi.csv` | Mensual | 2018–presente |

El pipeline entrena 5 tipos de modelos sobre 4 series IVF, con distintos métodos de backfill para los indicadores pre-2018. Incluye variable dummy de COVID-19 y conjunto de prueba post-pandemia.

---

## Configuración del entorno

- Python 3.10+
- Git

```bash
git clone <url-del-repo>
cd paper_turismo
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
pip install -r requirements.txt
```

---

## Ejecutar el pipeline

> **Importante:** siempre activa el entorno virtual antes de usar `dvc`.

```bash
source .venv/Scripts/activate   # Windows bash
dvc repro
```

`dvc repro` detecta qué etapas necesitan re-ejecutarse y salta las que ya están al día.

---

## Experimentos del paper

El paper compara 5 arquitecturas de modelo × 3 métodos de backfill = 15 experimentos. El script de automatización los corre secuencialmente:

```bash
# Correr todos los experimentos y generar figuras de comparación
python src/run_all_experiments.py

# Vista previa sin ejecutar nada
python src/run_all_experiments.py --dry-run

# Correr solo un subconjunto
python src/run_all_experiments.py --only xgb_linear,mlp_linear
```

Después del run completo, ver todos los experimentos con:
```bash
dvc exp show
```

### Matriz de experimentos

| Experimento | Modelo | Backfill |
|---|---|---|
| `xgb_zero` | XGBoost | zero |
| `xgb_linear` | XGBoost | linear |
| `xgb_xgb_backcast` | XGBoost | xgboost_backcast |
| `mlp_zero` / `mlp_linear` / `mlp_xgb_backcast` | MLP | — |
| `gru_zero` / `gru_linear` / `gru_xgb_backcast` | GRU | — |
| `cnngru_*` | CNN-GRU | — |
| `rescnngru_*` | Res-CNN-GRU | — |

### Estructura de salidas

```
metrics/{exp_name}/metrics_ivf_*.json   ← 15 carpetas × 4 series
plots/{exp_name}/                        ← 15 carpetas × todos los plots
plots/general/
  ivf_overview.png          (todas las series IVF con banda COVID)
  model_comparison.png      (5 modelos × 4 series, backfill=linear)
  backfill_impact.png       (5 modelos × 3 métodos de backfill)
```

---

## Arquitecturas de modelo

| Tipo | Descripción | Archivo |
|---|---|---|
| `xgboost` | XGBoost con características tabulares de rezago | `src/models/xgboost_model.py` |
| `mlp` | MLP (FC 128→64→1, sin secuencias) | `src/models/mlp_model.py` |
| `gru` | GRU bidimensional (seq_len=16, hidden=128, layers=2) | `src/models/gru_model.py` |
| `cnn_gru` | Conv1D × 3 + GRU (sin residuales) | `src/models/cnn_gru_model.py` |
| `res_cnn_gru` | ResConv × 5 + GRU × 3 (arquitectura completa) | `src/models/res_cnn_gru_model.py` |

### Métodos de backfill para indicadores pre-2018

| Método | Descripción |
|---|---|
| `zero` | Rellena con ceros (baseline) |
| `linear` | Regresión lineal extrapolada hacia atrás (recomendado) |
| `xgboost_backcast` | XGBoost entrenado sobre la serie invertida |
| `seasonal_mean` | Media del mismo trimestre 2018–2025 |
| `seasonal_naive` | Valor del mismo trimestre del año más próximo |

---

## Estructura del proyecto

```
paper_turismo/
├── data/
│   ├── raw/              # CSVs originales del INEGI (en git)
│   ├── processed/        # CSVs limpios + indicadores con backcast
│   ├── features/         # Matrices de características por serie IVF
│   ├── forecasts/        # Pronósticos futuros por serie
│   └── shap/             # Valores SHAP (.npz) y nombres de features (.json)
├── models/               # Modelos entrenados y metadatos de división
├── metrics/              # Métricas por experimento (carpetas {exp_name}/)
├── plots/                # Gráficas por experimento (carpetas {exp_name}/) + general/
├── src/
│   ├── models/           # Registro de modelos
│   │   ├── base.py
│   │   ├── xgboost_model.py
│   │   ├── mlp_model.py
│   │   ├── gru_model.py
│   │   ├── cnn_gru_model.py
│   │   ├── res_cnn_gru_model.py
│   │   └── __init__.py
│   ├── clean_*.py
│   ├── features.py / features_ivf_multi.py
│   ├── backcast_indicadores.py
│   ├── train.py / evaluate.py / interpret.py / forecast.py
│   ├── compare_shap.py
│   ├── archive_experiment.py      # Archiva resultados del experimento actual
│   ├── run_all_experiments.py     # Automatiza los 15 experimentos del paper
│   ├── plot_ivf_overview.py
│   ├── plot_model_comparison.py
│   └── plot_backfill_impact.py
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

---

## Agregar un nuevo tipo de modelo

1. Crea `src/models/mi_modelo.py` implementando `BaseForecaster` (ver `src/models/base.py`)
2. Regístralo en `src/models/__init__.py`
3. Agrega sus hiperparámetros en `params.yaml` bajo `model:`
4. Cambia `model.type: mi_modelo` en `params.yaml`
5. Ejecuta `dvc repro`
