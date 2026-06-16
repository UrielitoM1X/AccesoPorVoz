import os
import re
import pandas as pd
import numpy as np
from scipy.io import wavfile

from audio_processor import extract_features


FEATURE_NAMES = [
    "F0",
    "F1",
    "F2",
    "E_grave",
    "E_baja",
    "E_media",
    "Fdom",
    "Centroide",
    "Relacion_E",
    "Relacion_F",
    "Relacion_Grave_Media"
]


SILABAS_VALIDAS = [
    "MA", "ME", "MI", "MO", "MU",
    "SA", "SE", "SI", "SO", "SU",
    "A", "E", "I", "O", "U"
]


def normalizar_audio(y):
    y = np.asarray(y, dtype=float)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    return y / (np.max(np.abs(y)) + 1e-12)


def extraer_silaba_nombre(nombre):
    nombre = str(nombre).upper()

    for silaba in sorted(SILABAS_VALIDAS, key=len, reverse=True):
        if re.search(rf"\b{silaba}\b", nombre):
            return silaba

    for silaba in sorted(SILABAS_VALIDAS, key=len, reverse=True):
        if silaba in nombre:
            return silaba

    return "DESCONOCIDA"


def crear_base_similitud(
    carpeta_audios="Muestras",
    salida_csv="data/processed/base_similitud.csv"
):
    registros = []

    os.makedirs(os.path.dirname(salida_csv), exist_ok=True)

    print("====================================")
    print(" CREANDO BASE DE SIMILITUD")
    print("====================================")
    print(f"Carpeta de audios: {carpeta_audios}")

    for root, dirs, files in os.walk(carpeta_audios):
        for file in files:
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)

                try:
                    print(f"Procesando: {path}")

                    sr, y = wavfile.read(path)
                    y = normalizar_audio(y)

                    features = extract_features(y, sr)

                    nombre = os.path.splitext(file)[0]
                    silaba = extraer_silaba_nombre(nombre)

                    registro = {
                        "Archivo": path,
                        "Nombre": nombre,
                        "Carpeta": os.path.basename(root),
                        "Silaba": silaba
                    }

                    for i, fname in enumerate(FEATURE_NAMES):
                        registro[fname] = float(features[i])

                    registros.append(registro)

                except Exception as e:
                    print(f"Error procesando {path}: {e}")

    df = pd.DataFrame(registros)

    if len(df) == 0:
        print("No se encontraron audios WAV.")
        return None

    df.to_csv(salida_csv, index=False)

    print("====================================")
    print(" BASE GENERADA CORRECTAMENTE")
    print("====================================")
    print(f"Total de audios procesados: {len(df)}")
    print(f"Guardado en: {salida_csv}")

    return salida_csv


def obtener_vecinos(features_nuevas, csv_base="data/processed/base_similitud.csv", top_n=5):
    if not os.path.exists(csv_base):
        return pd.DataFrame()

    df = pd.read_csv(csv_base)

    if len(df) == 0:
        return pd.DataFrame()

    X_base = df[FEATURE_NAMES].values.astype(float)
    x = np.asarray(features_nuevas, dtype=float)

    medias = np.mean(X_base, axis=0)
    desv = np.std(X_base, axis=0) + 1e-9

    X_base_norm = (X_base - medias) / desv
    x_norm = (x - medias) / desv

    distancias = np.linalg.norm(X_base_norm - x_norm, axis=1)

    df["Distancia"] = distancias
    df = df.sort_values("Distancia", ascending=True)

    return df.head(top_n)


def clasificar_silaba_por_vecinos(features_nuevas, csv_base="data/processed/base_similitud.csv", top_n=5):
    vecinos = obtener_vecinos(features_nuevas, csv_base, top_n)

    if vecinos.empty:
        return "DESCONOCIDA", vecinos

    votos = {}

    for _, row in vecinos.iterrows():
        silaba = row["Silaba"]
        distancia = row["Distancia"]

        peso = 1 / (distancia + 1e-9)

        if silaba not in votos:
            votos[silaba] = 0

        votos[silaba] += peso

    silaba_final = max(votos, key=votos.get)

    return silaba_final, vecinos