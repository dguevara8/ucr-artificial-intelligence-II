# Proyecto 2 — Predicción de Demanda Eléctrica con Agente Inteligente

Predicción de la demanda eléctrica a corto plazo (15 minutos adelante) utilizando modelos de regresión y un agente inteligente basado en datos reales del sistema eléctrico.

---

## Estructura del repositorio

```text
├── OperacionTiempoReal.txt
├── OperacionTiempoReal2.txt
├── OperacionTiempoReal3.txt
├── OperacionTiempoReal4.txt
├── OperacionTiempoReal5.txt
├── Preprocesamiento.py        # Limpieza y preparación de datos
├── Modelos.py                 # Entrenamiento, evaluación y predicción
├── Visualizacion.py           # Generación de gráficas
├── Agente.py                  # Agente inteligente
├── datos_limpios.csv          # Dataset final limpio
├── demanda_historica.png
├── predicciones_vs_real.png
├── comparacion_metricas.png
├── ProyectoIA_C23562.pdf      # Documentación del proyecto
├── MODELCARD.md
└── README.md
```

---

## Descripción del problema

El objetivo del proyecto es predecir la demanda eléctrica del sistema en el siguiente intervalo de 15 minutos utilizando datos históricos reales.

El sistema permite:

* Analizar el comportamiento de la demanda
* Predecir valores futuros
* Comparar modelos de aprendizaje automático
* Generar recomendaciones operativas

---

## Preprocesamiento (`Preprocesamiento.py`)

El pipeline de procesamiento incluye:

1. Carga de múltiples archivos de datos.
2. Limpieza de valores faltantes o inconsistentes.
3. Detección de huecos temporales.
4. Conversión de variables de tiempo:

   * Hora decimal
   * Día de la semana
   * Indicador de fin de semana
5. Generación de variables de rezago:

   * Lag 1 (15 minutos)
   * Lag 2 (30 minutos)
   * Lag 4 (60 minutos)

El resultado final es un dataset estructurado listo para entrenamiento.

---

## Dataset

El dataset final se genera tras el preprocesamiento de los datos originales.

| Característica       |      Valor |
| -------------------- | ---------: |
| Registros originales |        436 |
| Registros útiles     |        411 |
| Variables de entrada |          9 |
| Variable objetivo    |          1 |
| Intervalo temporal   | 15 minutos |

El archivo utilizado es:

```text
datos_limpios.csv
```

---

## Entrenamiento (`Modelos.py`)

Se entrenaron dos modelos de regresión:

* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)

Se utilizó una división secuencial para respetar la naturaleza temporal de los datos:

| Conjunto      | Porcentaje | Registros |
| ------------- | ---------: | --------: |
| Entrenamiento |        80% |      ~328 |
| Prueba        |        20% |       ~83 |

Se aplicó escalado de variables para mejorar el desempeño de los modelos.

---

## Resultados

| Modelo | RMSE (MW) | MAE (MW) |     R² |
| ------ | --------: | -------: | -----: |
| KNN    |     79.03 |    64.22 | 0.9147 |
| SVR    |    192.49 |   169.76 | 0.4942 |

---

## Mejor modelo

El mejor modelo fue **KNN (K-Nearest Neighbors)**.

| Métrica | Valor  |
| ------- | ------ |
| RMSE    | 79.03  |
| MAE     | 64.22  |
| R²      | 0.9147 |

Este modelo presenta:

* Menor error de predicción
* Mejor ajuste a los datos
* Mejor seguimiento de la curva real

---

## Visualización (`Visualizacion.py`)

Se generaron las siguientes gráficas:

* Demanda histórica
* Predicciones vs valores reales
* Comparación de métricas entre modelos

Estas permiten interpretar visualmente el desempeño del sistema.

---

## Agente Inteligente (`Agente.py`)

El agente implementa un ciclo de razonamiento:

1. Analiza la intención del usuario
2. Selecciona la herramienta adecuada
3. Ejecuta el modelo correspondiente
4. Genera una respuesta en lenguaje natural

### Funcionalidades

* Predicción de demanda
* Comparación de modelos
* Estadísticas del sistema
* Selección automática del mejor modelo

---

## Ejemplo de uso

**Pregunta:**

```text
¿Cuál será la demanda en los próximos 15 minutos?
```

**Respuesta:**

* Predicción de demanda futura
* Tendencia (sube o baja)
* Margen de error
* Recomendación operativa

---

## Reproducibilidad

Para reproducir el proyecto completo:

```bash
python Preprocesamiento.py
python Modelos.py
python Visualizacion.py
python Agente.py
```

---

## Requisitos

Instalar dependencias necesarias:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Limitaciones

* Dataset pequeño
* Huecos en la serie temporal
* No incluye variables externas (clima, feriados)
* No se optimizaron hiperparámetros exhaustivamente

---

## Mejoras futuras

* Incorporar más datos históricos
* Añadir variables externas
* Implementar modelos avanzados (Random Forest, XGBoost)
* Optimización de hiperparámetros
