"""
=============================================================
  PREPROCESAMIENTO DE DATOS - CENCE / ICE
  Proyecto IE0435 - Inteligencia Artificial Aplicada
=============================================================

Combina archivos OperacionTiempoReal*.txt, ordena la serie temporal,
elimina duplicados por fecha, detecta/corrige anomalías y guarda
datos_limpios.csv para los modelos.
"""

from pathlib import Path
import re

import pandas as pd


PATRON_ARCHIVOS = "OperacionTiempoReal*"
ARCHIVO_SALIDA = "datos_limpios.csv"
FRECUENCIA_ESPERADA = "15min"

COLUMNAS_REQUERIDAS = [
    "demanda",
    "regulacionAbajo",
    "fecha",
    "ace",
    "frecuencia",
    "intercambioSur",
    "intercambioNorte",
    "primaria",
    "regulacionArriba",
]


def clave_orden_natural(ruta: Path):
    """Ordena OperacionTiempoReal2 antes de OperacionTiempoReal10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", ruta.name)]


def buscar_archivos() -> list[Path]:
    archivos = sorted(Path(".").glob(PATRON_ARCHIVOS), key=clave_orden_natural)
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos '{PATRON_ARCHIVOS}' en {Path.cwd()}."
        )
    return archivos


def leer_archivo(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"{ruta.name} no contiene las columnas requeridas: {faltantes}")

    df = df[COLUMNAS_REQUERIDAS].copy()
    df["archivo_origen"] = ruta.name
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    columnas_numericas = [c for c in COLUMNAS_REQUERIDAS if c != "fecha"]
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    filas_invalidas = df["fecha"].isna().sum()
    if filas_invalidas:
        print(f"  Advertencia: {ruta.name} tiene {filas_invalidas} fechas inválidas; se omiten.")
        df = df.dropna(subset=["fecha"])

    return df


def corregir_anomalias_iqr(df: pd.DataFrame) -> pd.DataFrame:
    q1 = df["demanda"].quantile(0.25)
    q3 = df["demanda"].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    df = df.copy()
    df["demanda_original"] = df["demanda"]
    df["es_anomalia"] = (
        (df["demanda"] < limite_inferior) | (df["demanda"] > limite_superior)
    ).astype(int)
    df["demanda"] = df["demanda"].clip(limite_inferior, limite_superior)

    print("\n" + "=" * 50)
    print("DETECCION Y CORRECCION DE ANOMALIAS")
    print("=" * 50)
    print(f"Q1:                 {q1:.2f} MW")
    print(f"Q3:                 {q3:.2f} MW")
    print(f"IQR:                {iqr:.2f} MW")
    print(f"Limite inferior:    {limite_inferior:.2f} MW")
    print(f"Limite superior:    {limite_superior:.2f} MW")
    print(f"Anomalias corregidas por recorte IQR: {df['es_anomalia'].sum()}")

    return df


def agregar_features_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hora"] = df["fecha"].dt.hour
    df["minuto"] = df["fecha"].dt.minute
    df["hora_decimal"] = df["hora"] + df["minuto"] / 60.0
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["fecha_siguiente_esperada"] = df["fecha"] + pd.Timedelta(FRECUENCIA_ESPERADA)
    df["gap_minutos"] = df["fecha"].diff().dt.total_seconds().div(60)
    return df


def main() -> pd.DataFrame:
    archivos = buscar_archivos()

    print("=" * 50)
    print("ARCHIVOS ENCONTRADOS")
    print("=" * 50)
    for archivo in archivos:
        print(f"  - {archivo.name}")
    print(f"\nTotal de archivos: {len(archivos)}")

    dataframes = []
    for archivo in archivos:
        df_temp = leer_archivo(archivo)
        dataframes.append(df_temp)
        print(f"  {archivo.name}: {len(df_temp)} filas leidas")

    df = pd.concat(dataframes, ignore_index=True)
    df = df.sort_values(["fecha", "archivo_origen"]).reset_index(drop=True)
    print(f"\nTotal de filas combinadas: {len(df)}")

    duplicados = df.duplicated(subset="fecha", keep=False).sum()
    if duplicados:
        print(f"\nSe encontraron {duplicados} registros con fecha duplicada.")
        print("Se conserva la ultima aparicion ordenada por nombre de archivo.")
        df = df.drop_duplicates(subset="fecha", keep="last").reset_index(drop=True)
    else:
        print("\nNo se encontraron fechas duplicadas entre archivos.")

    df = agregar_features_tiempo(df)
    df = corregir_anomalias_iqr(df)

    nulos_antes = df.isnull().sum().sum()
    df = df.dropna(subset=["demanda", "fecha", "frecuencia", "intercambioSur", "intercambioNorte", "primaria"])
    print(f"\nValores nulos eliminados en columnas criticas: {nulos_antes - df.isnull().sum().sum()}")

    pasos_esperados = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq=FRECUENCIA_ESPERADA)
    faltantes = pasos_esperados.difference(df["fecha"])
    print("\n" + "=" * 50)
    print("COBERTURA TEMPORAL")
    print("=" * 50)
    print(f"Rango de fechas: {df['fecha'].min()} -> {df['fecha'].max()}")
    print(f"Registros esperados si la serie fuera continua: {len(pasos_esperados)}")
    print(f"Registros disponibles despues de limpieza:       {len(df)}")
    print(f"Intervalos faltantes detectados:                 {len(faltantes)}")
    if len(faltantes) > 0:
        print("Nota: los modelos no crearan rezagos atravesando estos huecos.")

    df.to_csv(ARCHIVO_SALIDA, index=False)

    print("\n" + "=" * 50)
    print("RESUMEN FINAL")
    print("=" * 50)
    print(f"Archivos combinados: {len(archivos)}")
    print(f"Archivo guardado: {ARCHIVO_SALIDA}")
    print(f"Total de registros: {len(df)}")
    print(f"Demanda minima: {df['demanda'].min():.2f} MW")
    print(f"Demanda maxima: {df['demanda'].max():.2f} MW")
    return df


if __name__ == "__main__":
    main()
