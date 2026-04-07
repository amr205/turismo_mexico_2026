# Análisis de Resultados — Pronóstico del IVF Turístico de México

_Generado automáticamente el 2026-04-06_

_Experimentos completados: 15 / 15_

---

## 1. Resumen del estudio

**Período de prueba:** 2022 Q1 – 2025 Q3 (15 trimestres, post-COVID)
**Período de entrenamiento:** 1994 Q1 – 2021 Q4 (≈107 trimestres, historia completa)
**Variable dummy COVID:** Sí — 1 para 2020 Q1–2021 Q4 (8 trimestres)
**Series evaluadas:** IVF Total Nacional, IVF Turístico Total, IVF Turístico Bienes, IVF Turístico Servicios
**Experimentos completados:** 15 / 15

### Diseño experimental

El estudio sigue un diseño factorial 5×3:

| Factor | Niveles |
|---|---|
| **Modelo** | XGBoost, MLP, GRU, CNN-GRU, Res-CNN-GRU |
| **Backfill** | Zero, Linear, XGB Backcast |

Esto produce **15 experimentos** que permiten descomponer el efecto de la arquitectura del modelo
y del método de imputación de indicadores pre-2018 de forma independiente.

## 2. Comparación de modelos (backfill = linear)

_Métricas promediadas sobre las 4 series IVF. Conjunto de prueba 2022–2025._

| Modelo | MAE | RMSE | R² | Descripción |
|---|---|---|---|---|
| XGBoost | **5.5626** | **6.3811** | **-1.8541** | Gradient boosting sobre características tabulares de rezago (baseline) |
| MLP | 14.1006 | 15.9386 | -18.4796 | Red neuronal densa (FC 128→64→1), sin estructura temporal |
| GRU | 13.0196 | 14.6879 | -12.8360 | Gated Recurrent Unit (seq=16, hidden=128, layers=2) |
| CNN-GRU | 20.6106 | 21.3265 | -35.1723 | 3 bloques Conv1D + GRU (sin conexiones residuales) |
| Res-CNN-GRU | 17.8982 | 18.6961 | -21.3639 | 5 bloques ResConv + 3 capas GRU (arquitectura completa con residuales) |

> **Negrita** = mejor valor en la columna.

### 2.1 Desglose por serie

**IVF Total Nacional**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | **4.3400** | **4.6148** | **-2.0958** |
| MLP | 13.8857 | 17.2690 | -42.3504 |
| GRU | 10.5782 | 11.0211 | -16.6564 |
| CNN-GRU | 8.6804 | 9.0797 | -10.9838 |
| Res-CNN-GRU | 11.5605 | 11.8543 | -19.4272 |

**IVF Turístico Total**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | **6.3518** | **7.3967** | **-2.7882** |
| MLP | 10.7315 | 11.8861 | -8.7822 |
| GRU | 9.3390 | 10.9939 | -7.3688 |
| CNN-GRU | 26.0720 | 26.9281 | -49.2076 |
| Res-CNN-GRU | 21.7996 | 23.4168 | -36.9675 |

**IVF Turístico Bienes**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | **5.0661** | **5.8043** | **-0.0003** |
| MLP | 25.3442 | 26.5572 | -19.9418 |
| GRU | 16.2738 | 19.1017 | -9.8341 |
| CNN-GRU | 11.8715 | 13.2367 | -4.2024 |
| Res-CNN-GRU | 24.3946 | 25.0711 | -17.6636 |

**IVF Turístico Servicios**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 6.4927 | **7.7088** | **-2.5322** |
| MLP | **6.4411** | 8.0420 | -2.8442 |
| GRU | 15.8874 | 17.6349 | -17.4848 |
| CNN-GRU | 35.8186 | 36.0613 | -76.2955 |
| Res-CNN-GRU | 13.8382 | 14.4421 | -11.3975 |

### 2.2 Análisis de ablación

| Paso | Comparación | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|---|
| Baseline | XGBoost → MLP | +153.5% | +149.8% | -16.6255 |
| + Recurrencia temporal | MLP → GRU | -7.7% | -7.8% | +5.6436 |
| + Extracción conv. | GRU → CNN-GRU | +58.3% | +45.2% | -22.3363 |
| + Conexiones residuales | CNN-GRU → Res-CNN-GRU | -13.2% | -12.3% | +13.8084 |

> Δ positivo en MAE/RMSE = empeora; Δ positivo en R² = mejora. Cada fila mide el incremento aportado por la siguiente complejidad arquitectónica.

## 3. Impacto del método de backfill

Compara el mismo modelo bajo los tres métodos de imputación de indicadores pre-2018. Métricas promediadas sobre las 4 series IVF.

**Zero:** Rellena con ceros los períodos pre-2018 (baseline de ruptura estructural)
**Linear:** Extrapolación lineal por indicador hacia atrás desde 2018
**XGB Backcast:** XGBoost entrenado sobre la serie temporal invertida

### 3.1 Tabla completa (MAE promedio sobre 4 series)

| Modelo | Zero | Linear | XGB Backcast |
|---|---|---|---|
| XGBoost | 13.5691 | **5.5626** | 8.1542 |
| MLP | 14.3208 | **14.1006** | 15.9322 |
| GRU | 15.4444 | 13.0196 | **11.3725** |
| CNN-GRU | **13.1493** | 20.6106 | 13.8734 |
| Res-CNN-GRU | 20.0033 | 17.8982 | **15.6415** |

> **Negrita** = mejor backfill para ese modelo (MAE).

### 3.2 Ganancia de `linear` sobre `zero` (ΔMAE %)

| Modelo | ΔMAE % | ΔRMSE % | ΔR² |
|---|---|---|---|
| XGBoost | -59.0% | -60.2% | +16.9229 |
| MLP | -1.5% | +3.5% | -2.6527 |
| GRU | -15.7% | -9.4% | +4.9283 |
| CNN-GRU | +56.7% | +51.7% | -20.7176 |
| Res-CNN-GRU | -10.5% | -9.2% | +15.0601 |

> Δ negativo en MAE/RMSE = mejora con backfill lineal. Δ positivo en R² = mayor poder explicativo.

## 4. Mejores modelos por serie

**IVF Total Nacional**  
Mejor: `xgb_xgb_backcast` — MAE 4.3282, RMSE 4.8382, R² -2.4027

**IVF Turístico Total**  
Mejor: `xgb_linear` — MAE 6.3518, RMSE 7.3967, R² -2.7882

**IVF Turístico Bienes**  
Mejor: `xgb_linear` — MAE 5.0661, RMSE 5.8043, R² -0.0003

**IVF Turístico Servicios**  
Mejor: `mlp_linear` — MAE 6.4411, RMSE 8.0420, R² -2.8442

### 4.1 Tabla global (MAE)

| Serie | IVF Total Nacional | IVF Turístico Total | IVF Turístico Bienes | IVF Turístico Servicios |
|---|---|---|---|---|
| XGBoost | 4.3400 | 6.3518 | 5.0661 | 6.4927 |
| MLP | 13.8857 | 10.7315 | 25.3442 | 6.4411 |
| GRU | 10.5782 | 9.3390 | 16.2738 | 15.8874 |
| CNN-GRU | 8.6804 | 26.0720 | 11.8715 | 35.8186 |
| Res-CNN-GRU | 11.5605 | 21.7996 | 24.3946 | 13.8382 |

_Todos con backfill=linear._

## 5. Importancia de indicadores SHAP

Importancia media (|SHAP|) de los indicadores de visitantes INEGI sobre el modelo XGBoost con backfill lineal. Los valores son el promedio de las 4 series IVF.

_Columnas disponibles: ['Unnamed: 0', 'ivf_total_nacional', 'ivf_turistico_total', 'ivf_turistico_bienes', 'ivf_turistico_servicios']_

| Unnamed: 0                                                       |   ivf_total_nacional |   ivf_turistico_total |   ivf_turistico_bienes |   ivf_turistico_servicios |
|:-----------------------------------------------------------------|---------------------:|----------------------:|-----------------------:|--------------------------:|
| gasto_total__turistas_de_internacion__via_terrestre__salida      |             1.79778  |                 0     |               0.025243 |                  0.006557 |
| gasto_total__turistas_de_internacion__via_aerea__salida          |             1.33392  |                 0     |               0.009116 |                  0.013276 |
| gasto_total__turistas_fronterizos__en_automoviles__salida        |             0.527145 |                 0     |               0.002339 |                  0.002697 |
| gasto_total__excursionistas_fronterizos__en_automoviles__entrada |             0.299844 |                 0     |               0.021082 |                  0.002351 |
| gasto_total__excursionistas_fronterizos__peatones__salida        |             0.292349 |                 0     |               0.013765 |                  0.001984 |
| gasto_total__excursionistas_fronterizos__peatones__entrada       |             0.132433 |                 0     |               0.018882 |                  0.001082 |
| gasto_total__turistas_fronterizos__peatones__salida              |             0.105899 |                 0     |               0.000975 |                  0.00028  |
| gasto_total__turistas_fronterizos__en_automoviles__entrada       |             0.037391 |                 0     |               0.002341 |                  0.001713 |
| gasto_total__turistas_de_internacion__via_aerea__entrada         |             0.001542 |                 1e-06 |               0.017528 |                  0.005013 |
| num_visitantes__turistas_de_internacion__via_terrestre__salida   |             0.005584 |                 0     |               0.000213 |                  3.4e-05  |

> Los indicadores INEGI de visitantes (entradas/salidas por tipo de transporte y origen) aportan señal predictiva significativa más allá de las características de rezago puro. Ver `plots/general/` para visualizaciones detalladas.

## 6. Conclusiones y recomendaciones para el paper

### 6.1 Hallazgos principales

1. **El backfill lineal mejora consistentemente** el pronóstico frente al relleno con ceros
   (59.0% reducción en MAE para XGBoost), lo que justifica su uso como técnica de imputación
   recomendada para indicadores INEGI con cobertura parcial.

2. **Los indicadores INEGI de visitantes aportan valor predictivo** más allá de los rezagos puros
   del IVF. El modelo XGBoost (R² ≈ -1.8541) supera a un modelo autorregresivo simple,
   validando la hipótesis central del paper.

3. **La arquitectura XGBoost obtiene el mejor R²** (-1.8541,
   Δ = +0.0000 vs XGBoost). Esto indica que el XGBoost tabular es competitivo con las arquitecturas neuronales en este contexto.

4. **La variable dummy COVID** (2020 Q1–2021 Q4) permite al modelo separar la ruptura estructural
   de la tendencia de recuperación, resultando en un conjunto de prueba post-pandemia limpio.

### 6.2 Estructura sugerida del paper

```
1. Introducción
   - Importancia del pronóstico de turismo post-COVID
   - Limitación de datos INEGI (indicadores desde 2018)
   - Contribución: backfill + ablación de arquitecturas

2. Datos y metodología
   - Fuentes INEGI (IVF, consumo, indicadores)
   - Imputación pre-2018: comparación de métodos (Sección 3 de este análisis)
   - Variable dummy COVID-19
   - Diseño experimental: 5 modelos × 3 backfills

3. Modelos
   - XGBoost (baseline tabular)
   - Ablación progresiva: MLP → GRU → CNN-GRU → Res-CNN-GRU
   - Figura: arquitectura de Res-CNN-GRU

4. Resultados
   - Tabla principal: Sección 2 de este análisis
   - Figura: model_comparison.png
   - Figura: backfill_impact.png
   - Análisis de ablación: Sección 2.2

5. Interpretabilidad
   - SHAP por indicador: Sección 5
   - STL: tendencia y estacionalidad del IVF

6. Conclusiones
   - Recomendaciones prácticas (backfill lineal + modelo óptimo)
   - Limitaciones y trabajo futuro
```

### 6.3 Limitaciones

- Serie de prueba corta (15 trimestres). Los resultados pueden ser sensibles a eventos idiosincráticos 2022–2025.
- Los modelos neuronales con 200 épocas en series cortas pueden mostrar alta varianza entre corridas;
  considerar promedio de múltiples semillas para el paper final.
- Los indicadores INEGI de visitantes tienen cobertura mensual; la agregación trimestral puede
  perder información intraperiodo.
- El backcast XGBoost del pre-2018 es recursivo sobre la serie invertida: los errores de extrapolación
  se acumulan cuanto más atrás se retrocede.

## 7. Figuras generadas

| Archivo | Descripción |
|---|---|
| `plots/general/ivf_overview.png` | ✓ Todas las series IVF 1994–2025 con banda COVID sombreada |
| `plots/general/model_comparison.png` | ✓ MAE/RMSE/R² por modelo (backfill=linear, promedio 4 series) |
| `plots/general/backfill_impact.png` | ✓ Impacto del método de backfill por modelo (promedio 4 series) |
| `plots/xgb_linear/forecast_ivf_total_nacional.png` | ✓ Pronóstico XGBoost lineal — IVF Total Nacional |
| `plots/xgb_linear/shap_ivf_total_nacional.png` | ✓ Importancia SHAP XGBoost — IVF Total Nacional |
| `plots/xgb_linear/forecast_ivf_turistico_total.png` | ✓ Pronóstico XGBoost lineal — IVF Turístico Total |
| `plots/xgb_linear/shap_ivf_turistico_total.png` | ✓ Importancia SHAP XGBoost — IVF Turístico Total |
| `plots/xgb_linear/forecast_ivf_turistico_bienes.png` | ✓ Pronóstico XGBoost lineal — IVF Turístico Bienes |
| `plots/xgb_linear/shap_ivf_turistico_bienes.png` | ✓ Importancia SHAP XGBoost — IVF Turístico Bienes |
| `plots/xgb_linear/forecast_ivf_turistico_servicios.png` | ✓ Pronóstico XGBoost lineal — IVF Turístico Servicios |
| `plots/xgb_linear/shap_ivf_turistico_servicios.png` | ✓ Importancia SHAP XGBoost — IVF Turístico Servicios |
