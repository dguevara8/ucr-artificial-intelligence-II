"""
=============================================================
  AGENTE INTELIGENTE DE PREDICCION - IE0435
  Proyecto IE0435 - Inteligencia Artificial Aplicada
=============================================================

Agente sencillo con herramientas para predecir con KNN, SVR,
comparar modelos y consultar estadisticas del sistema.
"""

import re

from Modelos import (
    FEATURES,
    df,
    mejor_modelo,
    predecir_con_knn,
    predecir_con_mejor_modelo,
    predecir_con_svr,
    obtener_estadisticas,
    rmse_knn,
    rmse_svr,
)


def _tendencia(prediccion: float, demanda_actual: float) -> tuple[float, str]:
    cambio = prediccion - demanda_actual
    if cambio > 5:
        return cambio, "subira"
    if cambio < -5:
        return cambio, "bajara"
    return cambio, "se mantendra estable"


def herramienta_predecir_knn(datos: dict) -> dict:
    prediccion = predecir_con_knn(datos)
    cambio, tendencia = _tendencia(prediccion, datos["lag_1"])
    return {
        "modelo": "KNN (K-Nearest Neighbors)",
        "prediccion_mw": prediccion,
        "demanda_actual_mw": round(datos["lag_1"], 2),
        "cambio_mw": round(cambio, 2),
        "tendencia": tendencia,
        "rmse_modelo": round(rmse_knn, 2),
    }


def herramienta_predecir_svr(datos: dict) -> dict:
    prediccion = predecir_con_svr(datos)
    cambio, tendencia = _tendencia(prediccion, datos["lag_1"])
    return {
        "modelo": "SVR (Support Vector Regression)",
        "prediccion_mw": prediccion,
        "demanda_actual_mw": round(datos["lag_1"], 2),
        "cambio_mw": round(cambio, 2),
        "tendencia": tendencia,
        "rmse_modelo": round(rmse_svr, 2),
    }


def herramienta_predecir_mejor(datos: dict) -> dict:
    if mejor_modelo == "KNN":
        return herramienta_predecir_knn(datos)
    return herramienta_predecir_svr(datos)


def herramienta_estadisticas() -> dict:
    return obtener_estadisticas()


def herramienta_comparar_modelos(datos: dict) -> dict:
    res_knn = herramienta_predecir_knn(datos)
    res_svr = herramienta_predecir_svr(datos)
    pred_mejor = res_knn["prediccion_mw"] if mejor_modelo == "KNN" else res_svr["prediccion_mw"]
    return {
        "knn": res_knn,
        "svr": res_svr,
        "modelo_recomendado": mejor_modelo,
        "prediccion_recomendada_mw": pred_mejor,
        "razon": f"Se recomienda {mejor_modelo} porque tiene el menor RMSE en el conjunto de prueba.",
    }


def analizar_intencion(pregunta: str) -> str:
    p = pregunta.lower()

    if re.search(r"compar|ambos|los dos|knn y svr|svr y knn|mejor modelo|cual es mejor|cu[aá]l es mejor", p):
        return "comparar"
    if re.search(r"estad[ií]stica|estadisticas|datos|info|estado|calidad|rango|minimo|m[ií]nimo|maximo|m[aá]ximo|promedio|cuantos|cu[aá]ntos|registros", p):
        return "estadisticas"
    if re.search(r"svr|support vector|vector", p):
        return "svr"
    if re.search(r"knn|vecinos|neighbors", p):
        return "knn"
    if re.search(r"predic|demand|proximo|pr[oó]ximo|siguiente|15 min|cu[aá]nto|como estara|c[oó]mo estar[aá]|sera|ser[aá]", p):
        return "mejor"

    return "desconocido"


def clasificar_nivel(pred: float) -> tuple[str, str]:
    if pred > 1800:
        return "ALTA, cercana a pico del sistema", "Mantener reserva de generacion disponible."
    if pred > 1500:
        return "MEDIA-ALTA, dentro del rango operativo normal", "Monitoreo normal del sistema."
    if pred > 1300:
        return "MEDIA, operacion estable", "No se requieren acciones especiales."
    return "BAJA, posible hora valle", "Evaluar reduccion de unidades si aplica."


def generar_respuesta(intencion: str, resultado: dict) -> str:
    if intencion in ["knn", "svr", "mejor"]:
        pred = resultado["prediccion_mw"]
        actual = resultado["demanda_actual_mw"]
        cambio = resultado["cambio_mw"]
        signo = "+" if cambio > 0 else ""
        nivel, accion = clasificar_nivel(pred)

        return f"""
PREDICCION DE DEMANDA - {resultado['modelo']}
{'-' * 52}
Demanda actual:          {actual:.1f} MW
Prediccion a 15 min:     {pred:.1f} MW
Cambio esperado:         {signo}{cambio:.1f} MW; la demanda {resultado['tendencia']}
Error tipico del modelo: +/- {resultado['rmse_modelo']:.1f} MW
Nivel de demanda:        {nivel}

Recomendacion:
{accion}
{'-' * 52}"""

    if intencion == "comparar":
        knn = resultado["knn"]
        svr = resultado["svr"]
        return f"""
COMPARACION DE MODELOS DE PREDICCION
{'-' * 52}
KNN:
  Prediccion:   {knn['prediccion_mw']:.1f} MW
  Tendencia:    La demanda {knn['tendencia']}
  RMSE:         +/- {knn['rmse_modelo']:.1f} MW

SVR:
  Prediccion:   {svr['prediccion_mw']:.1f} MW
  Tendencia:    La demanda {svr['tendencia']}
  RMSE:         +/- {svr['rmse_modelo']:.1f} MW

Modelo recomendado:       {resultado['modelo_recomendado']}
Prediccion recomendada:   {resultado['prediccion_recomendada_mw']:.1f} MW
Razon: {resultado['razon']}
{'-' * 52}"""

    if intencion == "estadisticas":
        s = resultado
        nota_huecos = (
            "Hay huecos temporales; los rezagos se calcularon sin cruzar esos huecos."
            if s["intervalos_faltantes"] > 0
            else "La serie no presenta huecos de 15 minutos en el rango disponible."
        )
        return f"""
ESTADISTICAS DEL SISTEMA
{'-' * 52}
Registros limpios:        {s['total_registros_limpios']}
Registros para modelos:   {s['total_registros_modelo']}
Periodo cubierto:         {s['fecha_inicio']} -> {s['fecha_fin']}
Intervalos faltantes:     {s['intervalos_faltantes']}
Anomalias corregidas:     {s['anomalias_corregidas']}

Demanda minima:           {s['demanda_minima']:.1f} MW
Demanda maxima:           {s['demanda_maxima']:.1f} MW
Demanda promedio:         {s['demanda_promedio']:.1f} MW

KNN: RMSE={s['rmse_knn']:.2f} MW, MAE={s['mae_knn']:.2f} MW, R2={s['r2_knn']:.4f}
SVR: RMSE={s['rmse_svr']:.2f} MW, MAE={s['mae_svr']:.2f} MW, R2={s['r2_svr']:.4f}
Mejor modelo:             {s['mejor_modelo']}

Nota: {nota_huecos}
Un R2 negativo indica que se necesitan mas datos historicos para generalizar mejor.
{'-' * 52}"""

    return """
No entendi completamente tu pregunta. Puedes preguntar:
- Cual sera la demanda en los proximos 15 minutos?
- Predice con KNN
- Predice con SVR
- Compara KNN y SVR
- Dame estadisticas de los datos
"""


def datos_actuales_desde_ultima_fila() -> dict:
    ultima = df.iloc[-1]
    return {feature: float(ultima[feature]) for feature in FEATURES}


def agente(pregunta: str, datos_actuales: dict | None = None) -> str:
    if datos_actuales is None:
        datos_actuales = datos_actuales_desde_ultima_fila()

    intencion = analizar_intencion(pregunta)
    print(f"[AGENTE] Intencion detectada: '{intencion}'")

    if intencion == "knn":
        resultado = herramienta_predecir_knn(datos_actuales)
    elif intencion == "svr":
        resultado = herramienta_predecir_svr(datos_actuales)
    elif intencion == "mejor":
        resultado = herramienta_predecir_mejor(datos_actuales)
    elif intencion == "comparar":
        resultado = herramienta_comparar_modelos(datos_actuales)
    elif intencion == "estadisticas":
        resultado = herramienta_estadisticas()
    else:
        return generar_respuesta("desconocido", {})

    return generar_respuesta(intencion, resultado)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("AGENTE DE PREDICCION DE DEMANDA ELECTRICA")
    print("Sistema Electrico Nacional - Costa Rica")
    print("=" * 55)

    preguntas_prueba = [
        "Cual sera la demanda en los proximos 15 minutos?",
        "Predice con SVR",
        "Compara KNN y SVR",
        "Dame estadisticas de los datos",
    ]

    for i, pregunta in enumerate(preguntas_prueba, 1):
        print(f"\nPREGUNTA {i}: {pregunta}")
        print(agente(pregunta))

    print("\nMODO INTERACTIVO - escribe 'salir' para terminar")
    while True:
        try:
            pregunta = input("\nQue deseas saber? -> ").strip()
            if pregunta.lower() in ["salir", "exit", "quit"]:
                print("Hasta luego.")
                break
            if pregunta:
                print(agente(pregunta))
        except KeyboardInterrupt:
            print("\nHasta luego.")
            break
