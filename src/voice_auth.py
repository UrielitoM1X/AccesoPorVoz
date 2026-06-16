import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from audio_processor import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_DB_PATH = os.path.join(BASE_DIR, "data", "processed", "usuarios_biometria.csv")

FEATURE_COLUMNS = [
    "F0", "F1", "F2", "E_grave", "E_baja", "E_media", 
    "Fdom", "Centroide", "Relacion_E", "Relacion_F", "Relacion_Grave_Media"
]

def registrar_usuario(nombre_usuario):
    """
    Extrae los vectores de características del audio grabado y los almacena en el CSV.
    """
    try:
        archivo_wav = os.path.join(BASE_DIR, "data", "raw", f"reg_{nombre_usuario}.wav")
        
        if not os.path.exists(archivo_wav):
            return False, f"No se encontró el archivo de grabación en {archivo_wav}"
            
        sr, y = wavfile.read(archivo_wav)
        y = np.asarray(y, dtype=float)
        if y.ndim > 1: 
            y = np.mean(y, axis=1)
        
        # Normalización estricta de energía para mitigar ruido ambiental
        y = y - np.mean(y)
        if np.max(np.abs(y)) > 0:
            y = y / (np.max(np.abs(y)) + 1e-12)
        
        features = extract_features(y, sr)
        
        # Validar si las características son viables (evitar NaNs por completo)
        if np.isnan(features).any() or np.isinf(features).any():
            return False, "La muestra contiene ruido inestable. Intenta grabar en un entorno más silencioso."
            
        nuevo_registro = {
            "Usuario": nombre_usuario,
            "Archivo_Origen": f"reg_{nombre_usuario}.wav"
        }
        for idx, col_name in enumerate(FEATURE_COLUMNS):
            nuevo_registro[col_name] = float(features[idx])
            
        os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
        
        if os.path.exists(AUTH_DB_PATH):
            df_db = pd.read_csv(AUTH_DB_PATH)
            df_db = df_db[df_db["Usuario"] != nombre_usuario]
            df_final = pd.concat([df_db, pd.DataFrame([nuevo_registro])], ignore_index=True)
        else:
            df_final = pd.DataFrame([nuevo_registro])
            
        df_final.to_csv(AUTH_DB_PATH, index=False, encoding="utf-8")
        return True, f"Huella vocal de '{nombre_usuario}' integrada correctamente."
        
    except Exception as e:
        return False, f"Fallo en el registro: {str(e)}"


def verificar_voz(path_audio, umbral=0.15):
    """
    Compara el audio de login usando la métrica de Distancia Coseno, 
    altamente robusta frente a variaciones de volumen y ruido.
    """
    try:
        if not os.path.exists(AUTH_DB_PATH):
            return False, None, 1.0, "No hay usuarios registrados en la base de datos."
            
        sr, y = wavfile.read(path_audio)
        y = np.asarray(y, dtype=float)
        if y.ndim > 1: 
            y = np.mean(y, axis=1)
            
        y = y - np.mean(y)
        if np.max(np.abs(y)) > 0:
            y = y / (np.max(np.abs(y)) + 1e-12)
        
        features_login = extract_features(y, sr)
        
        df_db = pd.read_csv(AUTH_DB_PATH)
        if df_db.empty:
            return False, None, 1.0, "La base de datos biométrica está vacía."
            
        X_base = df_db[FEATURE_COLUMNS].values.astype(float)
        x_instancia = np.asarray(features_login, dtype=float)
        
        distancias_coseno = []
        
        # Calcular la Distancia Coseno para cada usuario en la BD
        for fila in X_base:
            # Reemplazar posibles NaNs locales por seguridad matemática
            fila = np.nan_to_num(fila)
            x_ins_clean = np.nan_to_num(x_instancia)
            
            norm_b = np.linalg.norm(fila)
            norm_i = np.linalg.norm(x_ins_clean)
            
            if norm_b == 0 or norm_i == 0:
                dist_c = 1.0  # Máxima distancia si el vector está vacío
            else:
                similitud_coseno = np.dot(fila, x_ins_clean) / (norm_b * norm_i)
                # La distancia coseno es 1 - similitud (Rango: 0 a 2. 0 significa idénticos)
                dist_c = 1.0 - similitud_coseno
                
            distancias_coseno.append(dist_c)
            
        idx_min = np.argmin(distancias_coseno)
        distancia_minima = distancias_coseno[idx_min]
        usuario_match = df_db.iloc[idx_min]["Usuario"]
        
        # --- BLOQUE DE DEPURACIÓN EN CONSOLA ---
        print("\n=== COMPULSA BIOMÉTRICA VALORES ===")
        print(f"Usuario evaluado: {usuario_match}")
        print(f"Features Base: {X_base[idx_min][:3]} ...")
        print(f"Features Login: {x_instancia[:3]} ...")
        print(f"DISTANCIA COSENO OBTENIDA: {distancia_minima:.4f}")
        print("====================================\n")
        
        # Un umbral coseno por defecto excelente y estricto es 0.05 a 0.15
        if distancia_minima <= umbral:
            return True, usuario_match, distancia_minima, "Firma acústica coincidente."
        else:
            return False, usuario_match, distancia_minima, f"Firma fuera de rango seguro."
            
    except Exception as e:
        return False, None, 1.0, f"Error en la verificación: {str(e)}"