"""
=============================================================
  MODELOS DE PREDICCION - KNN y SVR
  Proyecto IE0435 - Inteligencia Artificial Aplicada
=============================================================

Entrena modelos de prediccion de demanda electrica a 15 minutos.
Los rezagos solo se crean cuando el punto anterior existe realmente,
para evitar fuga de informacion en huecos de la serie temporal.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


warnings.filterwarnings("ignore")

ARCHIVO_DATOS = "datos_limpios.csv"
FRECUENCIA = pd.Timedelta(minutes=15)

FEATURES_BASE = [
    "hora_decimal",
    "dia_semana",
    "es_fin_semana",
    "frecuencia",
    "intercambioSur",
    "intercambioNorte",
    "primaria",
]
LAGS = {"lag_1": 1, "lag_2": 2, "lag_4": 4}
FEATURES = FEATURES_BASE + list(LAGS.keys())


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVO_DATOS)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


def crear_lags_y_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for nombre_lag, pasos in LAGS.items():
        df[nombre_lag] = df["demanda"].shift(pasos)
        diferencia = df["fecha"] - df["fecha"].shift(pasos)
        df.loc[diferencia != pasos * FRECUENCIA, nombre_lag] = np.nan

    df["target"] = df["demanda"].shift(-1)
    siguiente = df["fecha"].shift(-1) - df["fecha"]
    df.loc[siguiente != FRECUENCIA, "target"] = np.nan

    return df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)


def evaluar(y_real, y_pred) -> dict:
    mse = mean_squared_error(y_real, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
    }


def entrenar_modelos(verbose: bool = False) -> dict:
    df_limpio = cargar_datos()
    df_modelo = crear_lags_y_target(df_limpio)

    if len(df_modelo) < 20:
        raise ValueError(
            "Hay muy pocos registros consecutivos para entrenar. "
            "Descarga mas dias completos o revisa los huecos temporales."
        )

    X = df_modelo[FEATURES]
    y = df_modelo["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    n_neighbors = min(3, len(X_train))
    scaler_knn = StandardScaler()
    X_train_knn = scaler_knn.fit_transform(X_train)
    X_test_knn = scaler_knn.transform(X_test)

    modelo_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    modelo_knn.fit(X_train_knn, y_train)
    y_pred_knn = modelo_knn.predict(X_test_knn)
    metricas_knn = evaluar(y_test, y_pred_knn)

    scaler_X_svr = StandardScaler()
    X_train_svr = scaler_X_svr.fit_transform(X_train)
    X_test_svr = scaler_X_svr.transform(X_test)

    scaler_y_svr = StandardScaler()
    y_train_svr = scaler_y_svr.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    modelo_svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    modelo_svr.fit(X_train_svr, y_train_svr)
    y_pred_svr_scaled = modelo_svr.predict(X_test_svr)
    y_pred_svr = scaler_y_svr.inverse_transform(
        y_pred_svr_scaled.reshape(-1, 1)
    ).ravel()
    metricas_svr = evaluar(y_test, y_pred_svr)

    mejor_modelo = "KNN" if metricas_knn["rmse"] < metricas_svr["rmse"] else "SVR"

    contexto = {
        "df_limpio": df_limpio,
        "df": df_modelo,
        "X_test": X_test,
        "y_test": y_test,
        "fechas_test": df_modelo.loc[X_test.index, "fecha"],
        "modelo_knn": modelo_knn,
        "modelo_svr": modelo_svr,
        "scaler_knn": scaler_knn,
        "scaler_X_svr": scaler_X_svr,
        "scaler_y_svr": scaler_y_svr,
        "y_pred_knn": y_pred_knn,
        "y_pred_svr": y_pred_svr,
        "metricas_knn": metricas_knn,
        "metricas_svr": metricas_svr,
        "mejor_modelo": mejor_modelo,
    }

    if verbose:
        imprimir_resumen(contexto, X_train, X_test)

    return contexto


def imprimir_resumen(ctx: dict, X_train=None, X_test=None) -> None:
    print("=" * 55)
    print("CARGANDO DATOS")
    print("=" * 55)
    print(f"Registros limpios disponibles: {len(ctx['df_limpio'])}")
    print(f"Registros utiles para modelos: {len(ctx['df'])}")
    print(f"Columnas: {list(ctx['df_limpio'].columns)}")

    if X_train is not None and X_test is not None:
        print(f"\nDatos de entrenamiento: {len(X_train)} registros")
        print(f"Datos de prueba:        {len(X_test)} registros")

    print(f"\nFeatures utilizadas ({len(FEATURES)}):")
    for feature in FEATURES:
        print(f"  - {feature}")

    print("\n" + "=" * 55)
    print("MODELO 1: KNN (K-Nearest Neighbors)")
    print("=" * 55)
    imprimir_metricas(ctx["metricas_knn"])

    print("\n" + "=" * 55)
    print("MODELO 2: SVR (Support Vector Regression)")
    print("=" * 55)
    imprimir_metricas(ctx["metricas_svr"])

    print("\n" + "=" * 55)
    print("COMPARACION DE MODELOS")
    print("=" * 55)
    print(f"{'Metrica':<30} {'KNN':>10} {'SVR':>10}")
    print("-" * 55)
    print(f"{'RMSE (MW)':<30} {ctx['metricas_knn']['rmse']:>10.2f} {ctx['metricas_svr']['rmse']:>10.2f}")
    print(f"{'MAE (MW)':<30} {ctx['metricas_knn']['mae']:>10.2f} {ctx['metricas_svr']['mae']:>10.2f}")
    print(f"{'R2':<30} {ctx['metricas_knn']['r2']:>10.4f} {ctx['metricas_svr']['r2']:>10.4f}")
    print("-" * 55)
    print(f"\nMejor modelo segun RMSE: {ctx['mejor_modelo']}")


def imprimir_metricas(m: dict) -> None:
    print(f"MSE:  {m['mse']:.2f}")
    print(f"RMSE: {m['rmse']:.2f} MW")
    print(f"MAE:  {m['mae']:.2f} MW")
    print(f"R2:   {m['r2']:.4f}")


CTX = entrenar_modelos(verbose=False)

df_limpio = CTX["df_limpio"]
df = CTX["df"]
y_test = CTX["y_test"]
fechas_test = CTX["fechas_test"]
y_pred_knn = CTX["y_pred_knn"]
y_pred_svr = CTX["y_pred_svr"]
modelo_knn = CTX["modelo_knn"]
modelo_svr = CTX["modelo_svr"]
scaler_knn = CTX["scaler_knn"]
scaler_X_svr = CTX["scaler_X_svr"]
scaler_y_svr = CTX["scaler_y_svr"]
metricas_knn = CTX["metricas_knn"]
metricas_svr = CTX["metricas_svr"]
mejor_modelo = CTX["mejor_modelo"]

rmse_knn = metricas_knn["rmse"]
rmse_svr = metricas_svr["rmse"]
mae_knn = metricas_knn["mae"]
mae_svr = metricas_svr["mae"]
r2_knn = metricas_knn["r2"]
r2_svr = metricas_svr["r2"]


def _dataframe_nuevo(datos_nuevos: dict) -> pd.DataFrame:
    faltantes = [f for f in FEATURES if f not in datos_nuevos]
    if faltantes:
        raise ValueError(f"Faltan features para predecir: {faltantes}")
    return pd.DataFrame([datos_nuevos])[FEATURES]


def predecir_con_knn(datos_nuevos: dict) -> float:
    X_nuevo = _dataframe_nuevo(datos_nuevos)
    X_nuevo_scaled = scaler_knn.transform(X_nuevo)
    prediccion = modelo_knn.predict(X_nuevo_scaled)[0]
    return round(float(prediccion), 2)


def predecir_con_svr(datos_nuevos: dict) -> float:
    X_nuevo = _dataframe_nuevo(datos_nuevos)
    X_nuevo_scaled = scaler_X_svr.transform(X_nuevo)
    pred_scaled = modelo_svr.predict(X_nuevo_scaled)
    prediccion = scaler_y_svr.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    return round(float(prediccion), 2)


def predecir_con_mejor_modelo(datos_nuevos: dict) -> float:
    if mejor_modelo == "KNN":
        return predecir_con_knn(datos_nuevos)
    return predecir_con_svr(datos_nuevos)


def obtener_estadisticas() -> dict:
    huecos = pd.date_range(df_limpio["fecha"].min(), df_limpio["fecha"].max(), freq="15min").difference(df_limpio["fecha"])
    return {
        "total_registros_limpios": len(df_limpio),
        "total_registros_modelo": len(df),
        "fecha_inicio": str(df_limpio["fecha"].min()),
        "fecha_fin": str(df_limpio["fecha"].max()),
        "intervalos_faltantes": len(huecos),
        "demanda_minima": round(df_limpio["demanda"].min(), 2),
        "demanda_maxima": round(df_limpio["demanda"].max(), 2),
        "demanda_promedio": round(df_limpio["demanda"].mean(), 2),
        "anomalias_corregidas": int(df_limpio.get("es_anomalia", pd.Series(dtype=int)).sum()),
        "rmse_knn": round(rmse_knn, 2),
        "rmse_svr": round(rmse_svr, 2),
        "mae_knn": round(mae_knn, 2),
        "mae_svr": round(mae_svr, 2),
        "r2_knn": round(r2_knn, 4),
        "r2_svr": round(r2_svr, 4),
        "mejor_modelo": mejor_modelo,
    }


if __name__ == "__main__":
    imprimir_resumen(CTX)

    print("\n" + "=" * 55)
    print("PRUEBA RAPIDA DE FUNCIONES PARA EL AGENTE")
    print("=" * 55)

    ultima = df.iloc[-1]
    datos_ejemplo = {feature: float(ultima[feature]) for feature in FEATURES}
    print(f"\nFecha referencia: {ultima['fecha']}")
    print(f"Demanda actual:   {ultima['demanda']:.2f} MW")
    print(f"Prediccion KNN:   {predecir_con_knn(datos_ejemplo):.2f} MW")
    print(f"Prediccion SVR:   {predecir_con_svr(datos_ejemplo):.2f} MW")
    print(f"Mejor modelo:     {mejor_modelo}")

    print("\nEstadisticas del sistema:")
    for k, v in obtener_estadisticas().items():
        print(f"  {k}: {v}")
