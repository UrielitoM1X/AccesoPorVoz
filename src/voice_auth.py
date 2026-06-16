import os
import joblib
import numpy as np
from scipy.io import wavfile

from audio_processor import extract_features
from recorder import record_audio


VOICE_DB_PATH = "models/voice_users.pkl"


def cargar_base_usuarios():
    if os.path.exists(VOICE_DB_PATH):
        return joblib.load(VOICE_DB_PATH)
    return {}


def guardar_base_usuarios(base):
    os.makedirs("models", exist_ok=True)
    joblib.dump(base, VOICE_DB_PATH)


def obtener_vector_voz(path_audio):
    sr, y = wavfile.read(path_audio)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(float)
    y = y / (np.max(np.abs(y)) + 1e-12)

    features = extract_features(y, sr)

    return np.asarray(features, dtype=float)


def registrar_usuario(nombre_usuario, duracion=3, num_muestras=3):
    vectores = []

    os.makedirs(f"data/usuarios/{nombre_usuario}", exist_ok=True)

    for i in range(num_muestras):
        archivo = f"data/usuarios/{nombre_usuario}/muestra_{i+1}.wav"
        path = record_audio(archivo, duration=duracion)

        if path is None:
            return False, "Error al grabar una muestra."

        vector = obtener_vector_voz(path)
        vectores.append(vector)

    vector_promedio = np.mean(vectores, axis=0)

    base = cargar_base_usuarios()
    base[nombre_usuario] = vector_promedio
    guardar_base_usuarios(base)

    return True, f"Usuario {nombre_usuario} registrado correctamente."


def verificar_voz(path_audio, umbral=1.8):
    base = cargar_base_usuarios()

    if len(base) == 0:
        return False, None, None, "No hay usuarios registrados."

    vector_nuevo = obtener_vector_voz(path_audio)

    mejor_usuario = None
    mejor_distancia = float("inf")

    for usuario, vector_guardado in base.items():
        distancia = np.linalg.norm(vector_nuevo - vector_guardado)

        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_usuario = usuario

    if mejor_distancia <= umbral:
        return True, mejor_usuario, mejor_distancia, "Acceso permitido."

    return False, mejor_usuario, mejor_distancia, "Acceso denegado."