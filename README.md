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

Para los datasets trimestrales, el pipeline entrena un modelo **XGBoost con características de rezago** y produce:
- Métricas de evaluación (MAE, RMSE, MAPE, R²)
- Gráficas de importancia de características con **SHAP**
- Descomposición de tendencia/estacionalidad con **STL**

---

## Resultados del modelo base (XGBoost)

| Dataset | MAE | RMSE | MAPE | R² | Período de prueba |
|---|---|---|---|---|---|
| Consumo turístico | 5.02 | 10.27 | 6.96% | 0.271 | 2019 Q1 – 2025 Q2 (26 trimestres) |
| Índice volumen físico | 8.29 | 11.96 | 10.21% | 0.212 | 2019 Q1 – 2025 Q2 (26 trimestres) |

---

## Estructura del proyecto

```
paper_turismo/
├── data/
│   ├── raw/              # CSVs originales del INEGI (en git)
│   ├── processed/        # CSVs limpios (generados por el pipeline)
│   └── features/         # Matrices de características con rezagos (generadas)
├── models/               # Modelos entrenados (.json) y metadatos de división
├── metrics/              # Métricas de evaluación (.json, seguidas por DVC)
├── plots/                # Todas las gráficas generadas
├── src/
│   ├── models/           # Registro de modelos (extensible a LSTM, etc.)
│   │   ├── base.py       # Interfaz BaseForecaster
│   │   ├── xgboost_model.py
│   │   └── __init__.py   # Registro MODEL_REGISTRY
│   ├── clean_file_consumo_turistico.py
│   ├── clean_file_indice_volumen_fisico.py
│   ├── clean_turismo_indicadores.py
│   ├── plot_timeseries.py
│   ├── features.py       # Ingeniería de características
│   ├── train.py          # Entrenamiento del modelo
│   ├── evaluate.py       # Evaluación y gráfica de pronóstico
│   └── interpret.py      # SHAP + descomposición STL
├── dvc.yaml              # Definición del pipeline (11 etapas)
├── params.yaml           # Hiperparámetros y configuración del modelo
└── requirements.txt
```

---

## Configuración del entorno

### Requisitos previos

- Python 3.10+
- Git

### Instalación

1. Clona el repositorio:
   ```bash
   git clone <url-del-repo>
   cd paper_turismo
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python -m venv .venv

   # Windows (bash/Git Bash)
   source .venv/Scripts/activate

   # Windows (CMD)
   .venv\Scripts\activate.bat

   # macOS / Linux
   source .venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecutar el pipeline

> **Importante:** siempre activa el entorno virtual antes de usar `dvc`.

```bash
source .venv/Scripts/activate   # Windows bash
dvc repro
```

`dvc repro` detecta qué etapas necesitan re-ejecutarse (porque cambiaron sus dependencias o parámetros) y salta las que ya están al día.

Para forzar la re-ejecución de todas las etapas:
```bash
dvc repro --force
```

Para re-ejecutar una etapa específica y sus descendientes:
```bash
dvc repro train_consumo
```

### Etapas del pipeline

```
raw CSVs
 ├── clean_consumo    → processed/consumo_turistico_inegi_clean.csv
 ├── clean_ivf        → processed/indice_volumen_fisico_inegi_clean.csv
 ├── clean_indicadores→ processed/turismo_indicadores_inegi_clean.csv
 └── plot_timeseries  → plots/consumo_turistico.png, plots/ivf_turismo.png

processed/ (trimestrales, ~131 filas)
 ├── features_consumo → features/features_consumo.csv
 └── features_ivf     → features/features_ivf.csv

features/
 ├── train_consumo    → models/xgboost_consumo.json
 └── train_ivf        → models/xgboost_ivf.json

models/
 ├── evaluate_consumo → metrics/metrics_consumo.json + plots/forecast_consumo.png
 ├── evaluate_ivf     → metrics/metrics_ivf.json + plots/forecast_ivf.png
 ├── interpret_consumo→ plots/shap_consumo.png + plots/stl_consumo.png
 └── interpret_ivf    → plots/shap_ivf.png + plots/stl_ivf.png
```

---

## Métricas y gráficas

Ver las métricas actuales del workspace:
```bash
dvc metrics show
```

Para explorar experimentos y comparar configuraciones, consulta [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Agregar un nuevo tipo de modelo

El registro de modelos en `src/models/` permite agregar nuevos modelos sin modificar el pipeline:

1. Crea `src/models/mi_modelo.py` implementando `BaseForecaster` (ver `src/models/base.py`)
2. Regístralo en `src/models/__init__.py`
3. Agrega sus hiperparámetros en `params.yaml` bajo `model:`
4. Cambia `model.type: mi_modelo` en `params.yaml`
5. Ejecuta `dvc repro`

Para redes neuronales (LSTM, etc.), `get_shap_explainer` debe usar `shap.DeepExplainer` o `shap.KernelExplainer` en lugar de `shap.TreeExplainer`.
