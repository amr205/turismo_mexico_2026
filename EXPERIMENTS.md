# Experimentos con DVC

Esta guía explica cómo funciona DVC en este proyecto, cómo explorar resultados, cómo crear nuevos experimentos, y cómo agregar nuevas arquitecturas de modelo.

> **Requisito:** activa siempre el entorno virtual antes de ejecutar comandos DVC.
> ```bash
> source .venv/Scripts/activate   # Windows bash
> ```

---

## Cómo funciona DVC

DVC funciona como un **Makefile inteligente para ciencia de datos**. Define un grafo de dependencias (DAG) donde cada etapa tiene:

- **`deps`** — archivos o scripts de los que depende
- **`params`** — parámetros de `params.yaml` que usa
- **`outs`** — archivos que produce
- **`metrics`** — archivos de métricas (caso especial de `outs`, permite comparación entre experimentos)

Cuando ejecutas `dvc repro`, DVC compara los hashes actuales de deps y params contra los guardados en `dvc.lock`, y **solo re-ejecuta las etapas afectadas**. Si nada cambió, no hace nada.

### El flujo del pipeline

```
Raw CSVs
  └── clean_consumo / clean_ivf / clean_indicadores
        └── features_consumo / features_ivf
              └── train_consumo / train_ivf
                    └── evaluate_consumo / evaluate_ivf
                          └── interpret_consumo / interpret_ivf
```

Cada etapa aguas abajo se re-ejecuta automáticamente si algo aguas arriba cambia.

### Dónde se guardan las métricas

Las métricas se escriben en `metrics/metrics_consumo.json` y `metrics/metrics_ivf.json` por `src/evaluate.py`. Son archivos JSON normales — lo especial es que `dvc.yaml` los declara como `metrics:`, lo que permite a DVC compararlos entre experimentos con `dvc exp show`.

### `dvc repro` vs `dvc exp run` — diferencia clave

| | `dvc repro` | `dvc exp run` |
|---|---|---|
| Crea commit Git | No | Sí (en rama interna `refs/exps/...`) |
| Guarda experimento comparable | No | Sí |
| Modifica `params.yaml` | No | No (usa override temporal) |
| Aparece en `dvc exp show` | No | Sí |

`dvc repro` solo ejecuta los scripts y actualiza `dvc.lock`. El workspace queda en estado "dirty" — debes hacer commit manualmente si quieres versionar ese estado.

### Flujo con `dvc repro` (edición manual de params.yaml)

```bash
# 1. Editar params.yaml manualmente
# 2. Reproducir solo las etapas afectadas
dvc repro
# 3. Commitear manualmente
git add params.yaml dvc.lock metrics/ plots/ models/
git commit -m "exp: max_depth=5, MAE=..."
```

### Comandos de gestión del pipeline

| Comando | Para qué sirve |
|---|---|
| `dvc repro` | Re-ejecutar solo etapas afectadas |
| `dvc repro --force` | Re-ejecutar todo |
| `dvc repro train_consumo` | Re-ejecutar una etapa y sus dependientes |
| `dvc status` | Ver qué etapas están desactualizadas |
| `dvc params diff` | Ver qué parámetros cambiaron desde el último commit |
| `dvc dag` | Visualizar el grafo del pipeline |

---

## Explorar resultados existentes

### Ver métricas del workspace actual

```bash
dvc metrics show
```

Salida de ejemplo:
```
Path                          mae     mape     n_test    r2     rmse
metrics\metrics_consumo.json  5.0235  6.9558   26        0.271  10.2673
metrics\metrics_ivf.json      8.2883  10.2147  26        0.212  11.963
```

### Ver todos los experimentos registrados

```bash
dvc exp show
```

Muestra una tabla con cada experimento, sus parámetros y métricas. El workspace actual aparece como `workspace`.

Para filtrar y mostrar solo las columnas que cambiaron entre experimentos:
```bash
dvc exp show --only-changed
```

### Comparar dos experimentos específicos

```bash
dvc exp diff nombre_exp_1 nombre_exp_2
```

Muestra las diferencias en parámetros y métricas entre los dos experimentos.

### Exportar resultados a CSV (útil para el artículo)

```bash
dvc exp show --csv > experiments_summary.csv
```

---

## Crear un nuevo experimento

Un experimento en DVC es una ejecución del pipeline con parámetros diferentes. DVC guarda el estado completo (parámetros, métricas, gráficas) sin crear ramas de git adicionales.

### Sintaxis general

```bash
dvc exp run --set-param <seccion>.<parametro>=<valor> --name "nombre_descriptivo"
```

### Ejemplos de experimentos

**Cambiar hiperparámetros del modelo:**
```bash
# Árboles más profundos
dvc exp run --set-param model.xgboost.max_depth=5 --name "depth5"

# Más estimadores
dvc exp run --set-param model.xgboost.n_estimators=200 --name "n200"

# Tasa de aprendizaje más baja
dvc exp run --set-param model.xgboost.learning_rate=0.05 --name "lr005"
```

**Cambiar ingeniería de características:**
```bash
# Más rezagos (incluye el mismo trimestre del año anterior y el anterior)
dvc exp run --set-param "features.lags=[1,2,4,8]" --name "lags_ext"

# Sin dummies de trimestre
dvc exp run --set-param features.add_quarter_dummies=false --name "sin_dummies"

# Diferente variable objetivo para consumo
dvc exp run --set-param target.consumo=consumo_privado_nacional --name "objetivo_privado"
```

**Cambiar tamaño del conjunto de prueba:**
```bash
# 30% de prueba en lugar de 20%
dvc exp run --set-param train.test_size=0.3 --name "test30"
```

**Cambiar tipo de modelo** (cuando se agregue un nuevo modelo al registro):
```bash
dvc exp run --set-param model.type=lstm --name "lstm_baseline"
```

### Ejecutar múltiples experimentos en grilla

```bash
dvc exp run \
  --set-param model.xgboost.max_depth=2,3,4,5 \
  --set-param model.xgboost.n_estimators=50,100,200 \
  --name "grid_{params}"
```

Esto ejecuta 12 combinaciones (4 × 3) y las registra con nombres automáticos.

---

## Gestionar experimentos

### Aplicar el mejor experimento al workspace

Cuando encuentres un experimento con buenas métricas, aplícalo al workspace principal:

```bash
dvc exp apply nombre_del_experimento
```

Esto actualiza `params.yaml` y los archivos de salida. Luego haz commit:

```bash
git add params.yaml dvc.lock metrics/ plots/
git commit -m "feat: adoptar configuracion de experimento nombre_del_experimento"
```

### Guardar un experimento como rama de git

Si quieres preservar un experimento como rama para revisión:

```bash
dvc exp branch nombre_del_experimento nombre_de_rama
git checkout nombre_de_rama
```

### Eliminar experimentos que ya no necesitas

```bash
dvc exp remove nombre_del_experimento
```

### Listar experimentos en formato compacto

```bash
dvc exp list
```

---

## Parámetros disponibles en `params.yaml`

Todos los valores ajustables se encuentran en `params.yaml`. Aquí están las secciones principales:

| Sección | Parámetro | Descripción |
|---|---|---|
| `model.type` | `xgboost` | Tipo de modelo a usar |
| `model.xgboost.n_estimators` | `100` | Número de árboles |
| `model.xgboost.max_depth` | `3` | Profundidad máxima de cada árbol |
| `model.xgboost.learning_rate` | `0.1` | Tasa de aprendizaje |
| `model.xgboost.subsample` | `0.8` | Fracción de muestras por árbol |
| `features.lags` | `[1,2,3,4]` | Rezagos trimestrales como características |
| `features.rolling_windows` | `[2,4]` | Ventanas de media móvil |
| `features.add_quarter_dummies` | `true` | Incluir variables dummy de trimestre |
| `features.add_trend` | `true` | Incluir tendencia lineal como característica |
| `target.consumo` | `consumo_turistico_interior_total` | Variable objetivo para dataset de consumo |
| `target.ivf` | `ivf_turistico_total` | Variable objetivo para dataset IVF |
| `train.test_size` | `0.2` | Proporción del conjunto de prueba (cronológico) |
| `interpret.stl_period` | `4` | Período estacional para STL (4 = trimestral) |
| `interpret.shap_max_display` | `10` | Número máximo de características en gráfica SHAP |

---

## Flujo de trabajo recomendado

```
1. dvc exp run --set-param ... --name "exp_v1"
2. dvc exp show --only-changed          # comparar con baseline
3. dvc exp run --set-param ... --name "exp_v2"
4. dvc exp show --only-changed          # elegir el mejor
5. dvc exp apply exp_v2                 # aplicar al workspace
6. git add params.yaml dvc.lock metrics/ plots/
7. git commit -m "feat: ..."
```

---

## Agregar una nueva arquitectura de modelo

El proyecto usa un registro de modelos (`src/models/`) que permite agregar nuevas arquitecturas sin tocar los scripts del pipeline. Son exactamente 3 pasos.

### Paso 1 — Crear `src/models/<nombre>_model.py`

Debe implementar `BaseForecaster` (definido en `src/models/base.py`), que exige cinco métodos:

| Método | Descripción |
|---|---|
| `fit(X_train, y_train)` | Entrena el modelo |
| `predict(X)` | Genera predicciones |
| `save(path)` | Serializa el modelo a disco |
| `load(path)` | Carga el modelo desde disco (classmethod) |
| `get_shap_explainer(X_background)` | Retorna `shap_values` sobre datos de fondo |

Para la elección del explainer SHAP:
- XGBoost → `shap.TreeExplainer` (exacto, rápido)
- Redes neuronales → `shap.DeepExplainer` (PyTorch/TF) o `shap.KernelExplainer` (cualquier modelo, más lento)

Ejemplo de estructura para un LSTM en PyTorch:

```python
# src/models/lstm_model.py
import numpy as np, json, shap, torch, torch.nn as nn
from .base import BaseForecaster

class _LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class LSTMForecaster(BaseForecaster):
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train):
        # X_train es 2D — necesita reshape a 3D para LSTM
        X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y_train, dtype=torch.float32)
        self.model = _LSTMNet(X_train.shape[1], self.params["hidden_size"], self.params["num_layers"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.params["epochs"]):
            optimizer.zero_grad()
            loss = loss_fn(self.model(X), y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).numpy()

    def save(self, path: str):
        torch.save({"state_dict": self.model.state_dict(), "params": self.params}, path)

    @classmethod
    def load(cls, path: str) -> "LSTMForecaster":
        data = torch.load(path, weights_only=False)
        return cls(data["params"])

    def get_shap_explainer(self, X_background):
        explainer = shap.KernelExplainer(self.predict, X_background[:20])
        return explainer.shap_values(X_background)
```

### Paso 2 — Registrar en `src/models/__init__.py`

```python
from .xgboost_model import XGBoostForecaster
from .lstm_model import LSTMForecaster          # agregar

MODEL_REGISTRY = {
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,                     # agregar
}
```

### Paso 3 — Agregar hiperparámetros en `params.yaml`

```yaml
model:
  type: xgboost   # cambiar a "lstm" para usar el nuevo modelo

  xgboost:
    ...

  lstm:            # agregar bloque
    hidden_size: 64
    num_layers: 2
    epochs: 100
    learning_rate: 0.001
```

### Ejecutar el experimento con el nuevo modelo

```bash
# Sin tocar params.yaml (recomendado para comparar directamente con XGBoost):
dvc exp run --set-param model.type=lstm --name "lstm_baseline"

# Variando hiperparámetros:
dvc exp run \
  --set-param model.type=lstm \
  --set-param model.lstm.hidden_size=128 \
  --name "lstm_h128"

# Comparar con el baseline XGBoost:
dvc exp show --only-changed
```
