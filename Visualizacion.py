"""
=============================================================
  HERRAMIENTA DE VISUALIZACION - IE0435
  Proyecto IE0435 - Inteligencia Artificial Aplicada
=============================================================

Genera graficas de demanda historica, predicciones contra valores
reales y comparacion de metricas.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from Modelos import (
    df_limpio,
    fechas_test,
    y_test,
    y_pred_knn,
    y_pred_svr,
    mae_knn,
    mae_svr,
    rmse_knn,
    rmse_svr,
)


def insertar_cortes_por_huecos(df: pd.DataFrame, max_gap_minutos: int = 15) -> pd.DataFrame:
    """Inserta filas NaN para que matplotlib no una dias separados."""
    df = df.sort_values("fecha").copy()
    filas = []
    anterior = None
    for _, fila in df.iterrows():
        if anterior is not None:
            gap = (fila["fecha"] - anterior["fecha"]).total_seconds() / 60
            if gap > max_gap_minutos:
                corte = fila.copy()
                corte["fecha"] = anterior["fecha"] + pd.Timedelta(minutes=max_gap_minutos)
                for col in df.columns:
                    if col != "fecha":
                        corte[col] = float("nan")
                filas.append(corte)
        filas.append(fila)
        anterior = fila
    return pd.DataFrame(filas)


def formatear_eje_tiempo():
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.xticks(rotation=45, ha="right")


def graficar_demanda_historica(guardar_como: str = "demanda_historica.png") -> str:
    datos = insertar_cortes_por_huecos(df_limpio[["fecha", "demanda"]])

    plt.figure(figsize=(11, 5))
    plt.plot(datos["fecha"], datos["demanda"], color="#2563EB", linewidth=1.8, label="Demanda real")
    plt.title("Demanda Electrica Nacional - Costa Rica", fontsize=13, fontweight="bold")
    plt.xlabel("Fecha y hora")
    plt.ylabel("Demanda (MW)")
    formatear_eje_tiempo()
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(guardar_como, dpi=150)
    plt.close()
    print(f"Grafica guardada: {guardar_como}")
    return guardar_como


def graficar_predicciones_vs_real(guardar_como: str = "predicciones_vs_real.png") -> str:
    datos = pd.DataFrame(
        {
            "fecha": fechas_test.values,
            "real": y_test.values,
            "knn": y_pred_knn,
            "svr": y_pred_svr,
        }
    )
    datos = insertar_cortes_por_huecos(datos)

    plt.figure(figsize=(11, 5))
    plt.plot(datos["fecha"], datos["real"], color="#111827", linewidth=2, marker="o", markersize=4, label="Demanda real")
    plt.plot(datos["fecha"], datos["knn"], color="#16A34A", linewidth=1.8, linestyle="--", marker="s", markersize=3, label="Prediccion KNN")
    plt.plot(datos["fecha"], datos["svr"], color="#DC2626", linewidth=1.8, linestyle="--", marker="^", markersize=3, label="Prediccion SVR")
    plt.title("Demanda real vs. predicciones en prueba", fontsize=13, fontweight="bold")
    plt.xlabel("Fecha y hora")
    plt.ylabel("Demanda (MW)")
    formatear_eje_tiempo()
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(guardar_como, dpi=150)
    plt.close()
    print(f"Grafica guardada: {guardar_como}")
    return guardar_como


def graficar_comparacion_metricas(guardar_como: str = "comparacion_metricas.png") -> str:
    metricas = ["RMSE", "MAE"]
    valores_knn = [rmse_knn, mae_knn]
    valores_svr = [rmse_svr, mae_svr]

    x = range(len(metricas))
    ancho = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - ancho / 2 for i in x], valores_knn, width=ancho, label="KNN", color="#16A34A")
    plt.bar([i + ancho / 2 for i in x], valores_svr, width=ancho, label="SVR", color="#DC2626")
    plt.xticks(list(x), metricas)
    plt.ylabel("MW")
    plt.title("Comparacion de errores entre modelos", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(guardar_como, dpi=150)
    plt.close()
    print(f"Grafica guardada: {guardar_como}")
    return guardar_como


def herramienta_visualizar(tipo: str = "predicciones") -> str:
    opciones = {
        "historica": graficar_demanda_historica,
        "predicciones": graficar_predicciones_vs_real,
        "metricas": graficar_comparacion_metricas,
    }
    if tipo not in opciones:
        return "Tipo de grafica no reconocido. Usa: historica, predicciones o metricas"
    return opciones[tipo]()


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 55)
    graficar_demanda_historica()
    graficar_predicciones_vs_real()
    graficar_comparacion_metricas()
    print("\nListo. 3 graficas generadas.")
