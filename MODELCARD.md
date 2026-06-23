# Model Card — Predicción de Demanda Eléctrica a Corto Plazo

## Información general

Este modelo fue desarrollado para predecir la demanda eléctrica del sistema en intervalos de 15 minutos utilizando datos reales de operación.

El objetivo principal es estimar el consumo futuro inmediato para apoyar la toma de decisiones operativas, como la planificación de generación y la gestión de reservas del sistema eléctrico.

El problema se formula como una tarea de **regresión supervisada**, donde se predice un valor numérico continuo (demanda en MW).

---

## Modelos evaluados

Se evaluaron dos modelos de regresión:

* K-Nearest Neighbors Regressor (KNN)
* Support Vector Regression (SVR)

Estos modelos fueron seleccionados por representar enfoques distintos:

* KNN: basado en similitud entre observaciones
* SVR: basado en maximización de margen con funciones kernel

---

### K-Nearest Neighbors (KNN)

KNN estima la demanda futura comparando el estado actual del sistema con registros históricos similares.

El modelo identifica los *k* casos más cercanos en el espacio de variables y calcula la predicción como una combinación de sus valores.

Este enfoque es especialmente útil cuando existen patrones repetitivos en el comportamiento de la demanda, como ciclos diarios.

Sin embargo, su desempeño depende de:

* La calidad del escalado de variables
* La selección del número de vecinos
* La densidad de datos disponibles

---

### Support Vector Regression (SVR)

SVR busca encontrar una función que aproxime la relación entre las variables de entrada y la demanda, permitiendo cierto margen de error.

En este proyecto se utilizó un kernel RBF, lo que permite capturar relaciones no lineales.

Este modelo es más robusto en presencia de ruido, pero puede presentar dificultades cuando:

* Los datos son escasos
* Existen discontinuidades (como huecos temporales)

---

## Modelo seleccionado

El mejor modelo fue **KNN**, debido a su mejor desempeño en las métricas de evaluación.

| Métrica | Valor    |
| ------- | -------- |
| RMSE    | 79.03 MW |
| MAE     | 64.22 MW |
| R²      | 0.9147   |

Este modelo fue seleccionado porque:

* Minimiza el error de predicción
* Sigue mejor la forma de la curva de demanda
* Se adapta adecuadamente a patrones locales del sistema

---

## Métricas de evaluación

Para evaluar los modelos se utilizaron tres métricas principales:

### RMSE (Root Mean Squared Error)

El RMSE mide la magnitud del error penalizando más los errores grandes.

En este contexto, indica cuántos MW se desvía típicamente la predicción respecto al valor real.

Es útil para evaluar la precisión general del modelo en términos energéticos.

---

### MAE (Mean Absolute Error)

El MAE mide el error promedio en valor absoluto.

A diferencia del RMSE, no penaliza tanto los errores grandes, por lo que proporciona una medida más interpretable del error típico.

---

### R² (Coeficiente de determinación)

El R² indica qué proporción de la variabilidad de la demanda es explicada por el modelo.

* Valores cercanos a 1 → buen ajuste
* Valores cercanos a 0 → bajo poder explicativo

---

## Datos utilizados

El modelo fue entrenado con datos reales del sistema eléctrico organizados como una serie temporal.

Características relevantes:

* Variables temporales (hora, día)
* Variables del sistema eléctrico (frecuencia, intercambios, generación)
* Variables de memoria (lags de demanda)

El uso de variables de rezago permite capturar la dependencia temporal del sistema.

---

## Evaluación

La evaluación se realizó respetando el orden temporal de los datos, evitando mezclar información futura en el entrenamiento.

Se utilizó una división:

* 80% entrenamiento
* 20% prueba (datos más recientes)

Esto permite simular un escenario real de predicción.

---

## Interpretación del modelo

El modelo KNN no genera una función explícita, sino que basa sus predicciones en ejemplos históricos similares.

Esto implica que:

* Las predicciones son locales
* El modelo depende fuertemente de la calidad y continuidad de los datos
* Puede adaptarse bien a cambios suaves en la demanda

---

## Limitaciones

El modelo presenta las siguientes limitaciones:

* Sensibilidad a la cantidad de datos disponibles
* Dependencia de la continuidad temporal
* Falta de variables externas (clima, eventos, estacionalidad)
* Posible degradación en presencia de huecos en la serie

Además, el uso de un conjunto de datos reducido limita la capacidad del modelo para generalizar a largo plazo.

---

## Ideas para mejoras futuras

* Incorporación de variables externas (temperatura, feriados)
* Aumento del tamaño del dataset
* Uso de modelos más avanzados (Random Forest)
* Optimización de hiperparámetros
