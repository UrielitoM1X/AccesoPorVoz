import os
import pandas as pd
from scipy.io import wavfile

from src.audio_processor import extract_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, "Muestras")

data_list = []

print("Iniciando proceso...")
print("Buscando muestras en:", base_path)

for genero_dir in ["Hombres", "Mujeres"]:
    folder_path = os.path.join(base_path, genero_dir)

    print("Revisando:", folder_path)

    if not os.path.exists(folder_path):
        print("No existe:", folder_path)
        continue

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".wav"):
            continue

        nombre_sin_ext = filename.replace(".wav", "").replace(".WAV", "").upper()
        partes = nombre_sin_ext.split("-")

        silaba = partes[-1]

        if silaba not in ["SA", "SE", "SI", "SO", "SU"]:
            print("No se detectó sílaba válida en:", filename)
            continue

        genero = "H" if genero_dir == "Hombres" else "M"
        sujeto = nombre_sin_ext

        path = os.path.join(folder_path, filename)

        try:
            sr, y = wavfile.read(path)
        except Exception as e:
            print("Error leyendo:", filename, e)
            continue

        features = extract_features(y, sr)

        data_list.append({
            "Archivo": filename,
            "Genero": genero,
            "Sujeto": sujeto,
            "Silaba": silaba,
            "F0": round(features[0], 2),
            "F1": round(features[1], 2),
            "F2": round(features[2], 2),
            "E_baja": round(features[3], 6),
            "E_media": round(features[4], 6),
            "Fdom": round(features[5], 2),
            "Centroide": round(features[6], 2),
            "Relacion_E": round(features[7], 6),
            "Relacion_F": round(features[8], 6)
        })

        print("Procesado:", filename, "->", silaba, genero)

df = pd.DataFrame(data_list)

output_dir = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "AnalisisBD.csv")
df.to_csv(output_path, index=False, encoding="utf-8", sep=";")

print("Audios procesados:", len(df))
print("Archivo generado:", output_path)