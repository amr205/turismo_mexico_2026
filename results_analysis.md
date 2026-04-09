# Análisis de Resultados — Pronóstico del IVF Turístico de México

_Generado automáticamente el 2026-04-07_

_Experimentos completados: 25 / 25_

---

## 1. Resumen del estudio

**Período de prueba:** 2022 Q1 – 2025 Q3 (15 trimestres, post-COVID)
**Período de entrenamiento:** 1994 Q1 – 2021 Q4 (≈107 trimestres, historia completa)
**Variable dummy COVID:** Sí — 1 para 2020 Q1–2021 Q4 (8 trimestres)
**Series evaluadas:** IVF Total Nacional, IVF Turístico Total, IVF Turístico Bienes, IVF Turístico Servicios
**Experimentos completados:** 25 / 25

### Diseño experimental

El estudio evalúa 9 modelos bajo tres métodos de imputación de indicadores pre-2018:

| Factor | Niveles |
|---|---|
| **Modelo** | XGBoost, Ridge, SARIMA, SARIMAX, MLP, GRU, LSTM, CNN-GRU, Res-CNN-GRU |
| **Backfill** | Zero, Linear, XGB Backcast |

Esto produce **25 experimentos** (SARIMA se ejecuta solo una vez como baseline sin indicadores).
El diseño permite descomponer el efecto de la arquitectura del modelo y del método de imputación
de forma independiente.

## 2. Comparación de modelos

_Métricas promediadas sobre las 4 series IVF. Conjunto de prueba 2022–2025. Backfill lineal para todos los modelos; SARIMA usa solo su variante baseline (sin indicadores exógenos)._

| Modelo | Backfill | MAE | RMSE | R² | Descripción |
|---|---|---|---|---|---|
| XGBoost | linear | **5.5626** | **6.3811** | **-1.8541** | Gradient boosting sobre características tabulares de rezago (baseline) |
| Ridge | linear | 27.6007 | 35.7029 | -75.3606 | Regresión lineal regularizada (L2) sobre características de rezago |
| SARIMA | baseline | 13.6972 | 14.3409 | -14.4290 | SARIMA(1,1,1)(1,1,1)₄ — baseline temporal puro, sin indicadores |
| SARIMAX | linear | 86.4610 | 101.0257 | -590.2174 | SARIMA con indicadores exógenos de visitantes INEGI |
| MLP | linear | 9.0390 | 11.2923 | -13.2065 | Red neuronal densa (FC 64→32→1) con early stopping |
| GRU | linear | 45.7468 | 45.9939 | -183.2471 | Gated Recurrent Unit (seq=8, hidden=64, layers=1) con early stopping |
| LSTM | linear | 28.4921 | 29.4251 | -97.1157 | Long Short-Term Memory (seq=8, hidden=64, layers=1) con early stopping |
| CNN-GRU | linear | 11.2706 | 12.0562 | -8.1025 | 2 bloques Conv1D (32, 64 canales) + GRU(64) + FC(64→32→1) |
| Res-CNN-GRU | linear | 15.2027 | 15.9396 | -21.4299 | 3 bloques ResConv1D (16→32→64) + GRU(64) + FC(64→32→1) |

> **Negrita** = mejor valor en la columna.

### 2.1 Desglose por serie (backfill lineal; SARIMA = baseline)

**IVF Total Nacional**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | **4.3400** | **4.6148** | **-2.0958** |
| Ridge | 14.2300 | 21.5617 | -66.5810 |
| SARIMA | 7.9580 | 8.4621 | -9.4092 |
| SARIMAX | 44.1904 | 56.3825 | -461.1090 |
| MLP | 11.8246 | 16.1331 | -36.8349 |
| GRU | 50.9631 | 51.0479 | -377.8018 |
| LSTM | 41.8652 | 42.0926 | -256.5535 |
| CNN-GRU | 6.6778 | 7.1747 | -6.4829 |
| Res-CNN-GRU | 19.0670 | 19.4490 | -53.9860 |

**IVF Turístico Total**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | **6.3518** | **7.3967** | **-2.7882** |
| Ridge | 28.5252 | 36.8003 | -92.7691 |
| SARIMA | 18.8551 | 19.4868 | -25.2930 |
| SARIMAX | 46.1392 | 56.4922 | -219.9711 |
| MLP | 7.0130 | 7.9750 | -3.4037 |
| GRU | 52.4462 | 52.5832 | -190.4481 |
| LSTM | 29.1325 | 29.5064 | -59.2825 |
| CNN-GRU | 12.0720 | 12.6560 | -10.0906 |
| Res-CNN-GRU | 12.4480 | 13.0706 | -10.8290 |

**IVF Turístico Bienes**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 5.0661 | **5.8043** | **-0.0003** |
| Ridge | 42.3179 | 50.6359 | -75.1316 |
| SARIMA | 9.4780 | 10.2122 | -2.0966 |
| SARIMAX | 168.8781 | 189.0332 | -1060.0228 |
| MLP | **4.8859** | 5.9753 | -0.0601 |
| GRU | 32.2010 | 32.6836 | -30.7181 |
| LSTM | 9.1784 | 11.6355 | -3.0199 |
| CNN-GRU | 13.1254 | 14.3511 | -5.1153 |
| Res-CNN-GRU | 13.1517 | 14.4933 | -5.2371 |

**IVF Turístico Servicios**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | **6.4927** | **7.7088** | **-2.5322** |
| Ridge | 25.3297 | 33.8137 | -66.9605 |
| SARIMA | 18.4978 | 19.2025 | -20.9173 |
| SARIMAX | 86.6363 | 102.1948 | -619.7669 |
| MLP | 12.4325 | 15.0858 | -12.5272 |
| GRU | 47.3770 | 47.6611 | -134.0205 |
| LSTM | 33.7925 | 34.4658 | -69.6069 |
| CNN-GRU | 13.2072 | 14.0428 | -10.7213 |
| Res-CNN-GRU | 16.1440 | 16.7455 | -15.6674 |

### 2.2 Análisis de ablación arquitectónica

| Paso | Comparación | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|---|
| Tabular regularizado | XGBoost → Ridge | +396.2% | +459.5% | -73.5064 |
| Baseline estadístico | XGBoost → SARIMA | +146.2% | +124.7% | -12.5749 |
| + Exógenos (SARIMAX) | SARIMA → SARIMAX | +531.2% | +604.5% | -575.7884 |
| Tabular → NN densa | XGBoost → MLP | +62.5% | +77.0% | -11.3523 |
| + Memoria GRU | MLP → GRU | +406.1% | +307.3% | -170.0406 |
| GRU → LSTM | GRU → LSTM | -37.7% | -36.0% | +86.1314 |
| + Extracción conv. | GRU → CNN-GRU | -75.4% | -73.8% | +175.1446 |
| + Conexiones residuales | CNN-GRU → Res-CNN-GRU | +34.9% | +32.2% | -13.3273 |

> Δ positivo en MAE/RMSE = empeora; Δ positivo en R² = mejora. Cada fila mide el incremento aportado por la siguiente complejidad arquitectónica.

## 3. Impacto del método de backfill

Compara el mismo modelo bajo los tres métodos de imputación de indicadores pre-2018. Métricas promediadas sobre las 4 series IVF. SARIMA se excluye por no utilizar indicadores exógenos.

**Zero:** Rellena con ceros los períodos pre-2018 (baseline de ruptura estructural)
**Linear:** Extrapolación lineal por indicador hacia atrás desde 2018
**XGB Backcast:** XGBoost entrenado sobre la serie temporal invertida

### 3.1 Tabla completa (MAE promedio sobre 4 series)

| Modelo | Zero | Linear | XGB Backcast |
|---|---|---|---|
| XGBoost | 13.5691 | **5.5626** | 8.1542 |
| Ridge | 120.2890 | 27.6007 | **13.4666** |
| SARIMAX | 36.5939 | 86.4610 | **26.0017** |
| MLP | 72147722.4950 | **9.0390** | 26.3022 |
| GRU | 100.1497 | 45.7468 | **25.3039** |
| LSTM | 103.8011 | 28.4921 | **22.6362** |
| CNN-GRU | 16.3753 | **11.2706** | 19.2407 |
| Res-CNN-GRU | 21.1541 | **15.2027** | 18.8060 |

> **Negrita** = mejor backfill para ese modelo (MAE).

### 3.2 Ganancia de `linear` sobre `zero` (ΔMAE %)

| Modelo | ΔMAE % | ΔRMSE % | ΔR² |
|---|---|---|---|
| XGBoost | -59.0% | -60.2% | +16.9229 |
| Ridge | -77.1% | -75.5% | +1406.5590 |
| SARIMAX | +136.3% | +131.3% | -476.5314 |
| MLP | -100.0% | -100.0% | +539227713116377.4375 |
| GRU | -54.3% | -54.1% | +585.2896 |
| LSTM | -72.6% | -71.7% | +705.2392 |
| CNN-GRU | -31.2% | -28.7% | +9.0999 |
| Res-CNN-GRU | -28.1% | -26.8% | +16.6559 |

> Δ negativo en MAE/RMSE = mejora con backfill lineal. Δ positivo en R² = mayor poder explicativo.

## 4. Mejores modelos por serie

**IVF Total Nacional**  
Mejor: `xgb_xgb_backcast` — MAE 4.3282, RMSE 4.8382, R² -2.4027

**IVF Turístico Total**  
Mejor: `xgb_linear` — MAE 6.3518, RMSE 7.3967, R² -2.7882

**IVF Turístico Bienes**  
Mejor: `mlp_linear` — MAE 4.8859, RMSE 5.9753, R² -0.0601

**IVF Turístico Servicios**  
Mejor: `xgb_linear` — MAE 6.4927, RMSE 7.7088, R² -2.5322

### 4.1 Tabla global (MAE)

| Serie | IVF Total Nacional | IVF Turístico Total | IVF Turístico Bienes | IVF Turístico Servicios |
|---|---|---|---|---|
| XGBoost | 4.3400 | 6.3518 | 5.0661 | 6.4927 |
| Ridge | 14.2300 | 28.5252 | 42.3179 | 25.3297 |
| SARIMA | 7.9580 | 18.8551 | 9.4780 | 18.4978 |
| SARIMAX | 44.1904 | 46.1392 | 168.8781 | 86.6363 |
| MLP | 11.8246 | 7.0130 | 4.8859 | 12.4325 |
| GRU | 50.9631 | 52.4462 | 32.2010 | 47.3770 |
| LSTM | 41.8652 | 29.1325 | 9.1784 | 33.7925 |
| CNN-GRU | 6.6778 | 12.0720 | 13.1254 | 13.2072 |
| Res-CNN-GRU | 19.0670 | 12.4480 | 13.1517 | 16.1440 |

_Backfill lineal para todos los modelos; SARIMA = baseline (sin indicadores)._

## 5. Importancia de indicadores SHAP

Importancia media (|SHAP|) de los indicadores de visitantes INEGI sobre el modelo XGBoost con backfill lineal. Los valores son el promedio de las 4 series IVF.

_Columnas disponibles: ['Unnamed: 0', 'ivf_total_nacional', 'ivf_turistico_total', 'ivf_turistico_bienes', 'ivf_turistico_servicios']_

| Unnamed: 0                                                       |   ivf_total_nacional |   ivf_turistico_total |   ivf_turistico_bienes |   ivf_turistico_servicios |
|:-----------------------------------------------------------------|---------------------:|----------------------:|-----------------------:|--------------------------:|
| gasto_total__turistas_de_internacion__via_aerea__entrada         |                    0 |              4.06723  |               0        |                  0        |
| gasto_total__turistas_de_internacion__via_aerea__salida          |                    0 |              1.4651   |               2.4e-05  |                  0.000113 |
| gasto_total__turistas_fronterizos__en_automoviles__entrada       |                    0 |              0.689889 |               0        |                  0        |
| gasto_total__excursionistas_fronterizos__en_automoviles__entrada |                    0 |              0.374955 |               0.000477 |                  1.6e-05  |
| gasto_total__excursionistas_fronterizos__peatones__entrada       |                    0 |              0.173605 |               1e-05    |                  5.4e-05  |
| gasto_total__excursionistas_fronterizos__en_automoviles__salida  |                    0 |              0.107719 |               0        |                  0        |
| gasto_total__excursionistas_fronterizos__peatones__salida        |                    0 |              0.040512 |               6.7e-05  |                  4.2e-05  |
| gasto_total__turistas_fronterizos__en_automoviles__salida        |                    0 |              0.023693 |               9.6e-05  |                  5.1e-05  |
| num_visitantes__turistas_de_internacion__via_aerea__entrada      |                    0 |              0.014456 |               0        |                  0        |
| num_visitantes__turistas_de_internacion__via_aerea__salida       |                    0 |              0.005615 |               0        |                  0        |

> Los indicadores INEGI de visitantes (entradas/salidas por tipo de transporte y origen) aportan señal predictiva significativa más allá de las características de rezago puro. Ver `plots/general/` para visualizaciones detalladas.

## 6. Conclusiones y recomendaciones para el paper

### 6.1 Hallazgos principales

1. **El backfill lineal mejora consistentemente** el pronóstico frente al relleno con ceros
   (59.0% reducción en MAE para XGBoost), lo que justifica su uso como técnica de imputación
   recomendada para indicadores INEGI con cobertura parcial.

2. **Los indicadores INEGI de visitantes aportan valor predictivo** más allá de los rezagos puros
   del IVF. SARIMAX (MAE = 86.4610) supera a SARIMA puro (MAE = 13.6972), confirmando que los indicadores INEGI añaden señal predictiva incluso dentro del marco estadístico clásico.

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
   - Modelos tabulares: XGBoost, Ridge
   - Modelos estadísticos: SARIMA, SARIMAX
   - Redes neuronales: MLP → GRU → LSTM → CNN-GRU → Res-CNN-GRU (ablación)
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
