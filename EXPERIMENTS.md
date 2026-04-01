# Experimentos con DVC

Esta guía explica cómo explorar los resultados de experimentos existentes y cómo crear nuevos experimentos usando DVC.

> **Requisito:** activa siempre el entorno virtual antes de ejecutar comandos DVC.
> ```bash
> source .venv/Scripts/activate   # Windows bash
> ```

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
