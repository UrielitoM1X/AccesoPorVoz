import streamlit as st
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile

from audio_processor import extract_features
from predictor import AudioPredictor
from recorder import record_audio
from voice_auth import registrar_usuario, verificar_voz
from similar_finder import clasificar_silaba_por_vecinos

st.set_page_config(page_title="Clasificador de Voz - ESCOM", layout="centered")

st.title("🎙️ Analizador de Voz en Tiempo Real")
st.markdown(
    "Proyecto de Ingeniería en IA: Clasificación de Género, Sílabas y Acceso por Voz"
)


def load_models():
    return AudioPredictor()


predictor = load_models()


if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

if "usuario" not in st.session_state:
    st.session_state["usuario"] = None


def normalizar_audio(x):
    x = np.asarray(x, dtype=float)

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    max_val = np.max(np.abs(x)) + 1e-12
    return x / max_val


def graficar_audio(y, sr):
    t = np.linspace(0, len(y) / sr, num=len(y))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y, linewidth=0.8)
    ax.set_ylim([-1, 1])
    ax.set_title("Forma de onda del audio")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)

    st.pyplot(fig)


def graficar_forma_onda_archivo(path_audio, titulo="Forma de onda del audio"):
    sr, y = wavfile.read(path_audio)
    y = normalizar_audio(y)

    t = np.linspace(0, len(y) / sr, num=len(y))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y, linewidth=0.8)
    ax.set_ylim([-1, 1])
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)

    st.pyplot(fig)


def pantalla_login():
    st.subheader("🔐 Acceso por reconocimiento de voz")

    st.info(
        'Para entrar, graba tu voz diciendo una frase como: "Mi voz es mi contraseña".'
    )

    opcion = st.radio(
        "Selecciona una opción:", ["Verificar acceso", "Registrar usuario"]
    )

    if opcion == "Registrar usuario":
        st.write("### Registrar nuevo usuario autorizado")

        nombre = st.text_input("Nombre del usuario")
        duracion = st.slider("Duración de cada muestra", 2, 5, 3)
        muestras = st.slider("Número de muestras", 1, 5, 3)

        if st.button("Registrar voz"):
            if nombre.strip() == "":
                st.error("Escribe un nombre de usuario.")
                return

            with st.spinner("Registrando muestras de voz..."):
                ok, mensaje = registrar_usuario(
                    nombre_usuario=nombre.strip(),
                    duracion=duracion,
                    num_muestras=muestras,
                )

            if ok:
                st.success(mensaje)
            else:
                st.error(mensaje)

    else:
        st.write("### Verificar acceso")

        duracion = st.slider("Duración de verificación", 2, 5, 3)
        umbral = st.slider("Umbral de similitud", 50.0, 1000.0, 180.0, 5.0)

        if st.button("🎙️ Grabar voz y verificar"):
            os.makedirs("data/auth", exist_ok=True)

            timestamp = time.time_ns()
            temp_path = f"data/auth/login_{timestamp}.wav"

            with st.spinner("Grabando voz..."):
                path = record_audio(temp_path, duration=duracion)

            if path is None:
                st.error("No se pudo grabar el audio.")
                return

            st.audio(path, format="audio/wav")

            with st.spinner("Verificando identidad..."):
                acceso, usuario, distancia, mensaje = verificar_voz(
                    path_audio=path, umbral=umbral
                )

            if acceso:
                st.session_state["autenticado"] = True
                st.session_state["usuario"] = usuario
                st.success(f"✅ {mensaje} Usuario reconocido: {usuario}")
                st.rerun()
            else:
                st.error(f"❌ {mensaje}")
                if usuario is not None:
                    st.write(f"Usuario más parecido: {usuario}")
                    st.write(f"Distancia obtenida: {distancia:.4f}")


def mostrar_similares():
    if "resultado" not in st.session_state:
        return

    st.divider()
    st.subheader("🔎 5 vecinos más cercanos del dataset")

    features_actuales = st.session_state["resultado"]["features"]
    silaba_predicha = st.session_state["resultado"]["silaba"]

    silaba_predicha, similares = clasificar_silaba_por_vecinos(
        features_actuales, csv_base="data/processed/base_similitud.csv", top_n=5
    )

    if similares.empty:
        st.warning(
            "No se encontró la base de similitud. Ejecuta primero: python src/crear_base_similitud.py"
        )
        return

    for num, (_, row) in enumerate(similares.iterrows(), start=1):
        with st.container():
            st.markdown(f"### #{num} Coincidencia: `{row['Nombre']}`")

            if "Carpeta" in row:
                st.write(f"**Carpeta:** {row['Carpeta']}")

            st.write(f"**Distancia:** {row['Distancia']:.4f}")

            if os.path.exists(row["Archivo"]):
                st.audio(row["Archivo"], format="audio/wav")

                graficar_forma_onda_archivo(
                    row["Archivo"], titulo=f"Forma de onda de {row['Nombre']}"
                )
            else:
                st.warning("No se encontró el archivo de audio.")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.write(f"**F0:** {row['F0']:.2f} Hz")
                st.write(f"**F1:** {row['F1']:.2f} Hz")
                st.write(f"**F2:** {row['F2']:.2f} Hz")
                st.write(f"**E grave:** {row['E_grave']:.6f}")

            with col_b:
                st.write(f"**E baja:** {row['E_baja']:.6f}")
                st.write(f"**E media:** {row['E_media']:.6f}")
                st.write(f"**Fdom:** {row['Fdom']:.2f} Hz")
                st.write(f"**Centroide:** {row['Centroide']:.2f}")

            with col_c:
                st.write(f"**Relación E:** {row['Relacion_E']:.6f}")
                st.write(f"**Relación F:** {row['Relacion_F']:.6f}")
                st.write(f"**Relación Grave/Media:** {row['Relacion_Grave_Media']:.6f}")

            st.divider()


def sistema_clasificador():
    top_col1, top_col2 = st.columns([1, 5])

    with top_col1:
        if st.button("🔒 Cerrar sesión"):
            st.session_state["autenticado"] = False
            st.session_state["usuario"] = None
            st.rerun()

    with top_col2:
        st.success(f"Usuario autenticado: {st.session_state['usuario']}")

    st.sidebar.header("Configuración")
    duration = st.sidebar.slider("Duración de grabación (seg)", 1, 5, 3)
    st.sidebar.write(f"Duración actual: {duration} segundos")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔴 Iniciar Grabación"):

            if "resultado" in st.session_state:
                del st.session_state["resultado"]

            st.write("🎬 Iniciando grabación...")

            with st.spinner("Grabando..."):
                os.makedirs("data/raw", exist_ok=True)

                timestamp = time.time_ns()
                temp_path = f"data/raw/live_capture_{timestamp}.wav"

                path = record_audio(temp_path, duration=duration)

                if path is None:
                    st.error("❌ Error al grabar audio")
                    st.stop()

                st.success("✅ Grabación terminada")
                st.audio(path, format="audio/wav")

            with st.spinner("Analizando frecuencias..."):
                sr, y = wavfile.read(path)
                y = normalizar_audio(y)

                graficar_audio(y, sr)

                features = extract_features(y, sr)

                f0 = features[0]
                f1 = features[1]
                f2 = features[2]

                e_grave = features[3]
                e_baja = features[4]
                e_media = features[5]

                fdom = features[6]
                centroide = features[7]

                relacion_e = features[8]
                relacion_f = features[9]
                relacion_grave_media = features[10]

                st.write(f"F0: {f0:.2f} Hz")
                st.write(f"F1: {f1:.2f} Hz")
                st.write(f"F2: {f2:.2f} Hz")
                st.write(f"Energía Grave: {e_grave:.6f}")
                st.write(f"Energía Baja: {e_baja:.6f}")
                st.write(f"Energía Media: {e_media:.6f}")
                st.write(f"Fdom: {fdom:.2f} Hz")
                st.write(f"Centroide: {centroide:.2f}")
                st.write(f"Relación E: {relacion_e:.6f}")
                st.write(f"Relación F: {relacion_f:.6f}")
                st.write(f"Relación Grave/Media: {relacion_grave_media:.6f}")

                genero = predictor.predict_genero(f0)

                silaba, vecinos = clasificar_silaba_por_vecinos(
                    features, csv_base="data/processed/base_similitud.csv", top_n=5
                )

                st.session_state["resultado"] = {
                    "genero": genero,
                    "silaba": silaba,
                    "f0": f0,
                    "f1": f1,
                    "f2": f2,
                    "e_grave": e_grave,
                    "e_baja": e_baja,
                    "e_media": e_media,
                    "fdom": fdom,
                    "centroide": centroide,
                    "relacion_e": relacion_e,
                    "relacion_f": relacion_f,
                    "relacion_grave_media": relacion_grave_media,
                    "archivo": path,
                    "features": features,
                }

    with col2:
        st.subheader("Predicción del Modelo")

        if "resultado" in st.session_state:
            res = st.session_state["resultado"]

            st.metric(label="Género Detectado", value=res["genero"])
            st.metric(label="Sílaba Detectada", value=res["silaba"])

            st.write(f"**F0:** {res['f0']:.2f} Hz")
            st.write(f"**F1:** {res['f1']:.2f} Hz")
            st.write(f"**F2:** {res['f2']:.2f} Hz")
            st.write(f"**Energía Grave:** {res['e_grave']:.6f}")
            st.write(f"**Energía Baja:** {res['e_baja']:.6f}")
            st.write(f"**Energía Media:** {res['e_media']:.6f}")
            st.write(f"**Fdom:** {res['fdom']:.2f} Hz")
            st.write(f"**Centroide:** {res['centroide']:.2f}")
            st.write(f"**Relación E:** {res['relacion_e']:.6f}")
            st.write(f"**Relación F:** {res['relacion_f']:.6f}")
            st.write(f"**Relación Grave/Media:** {res['relacion_grave_media']:.6f}")
            st.write(f"**Archivo:** `{res['archivo']}`")

        else:
            st.info("Presiona el botón para analizar un audio.")

    mostrar_similares()


if not st.session_state["autenticado"]:
    pantalla_login()
else:
    sistema_clasificador()
