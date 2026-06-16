import streamlit as st
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Importaciones locales de tu arquitectura
from audio_processor import extract_features
from predictor import AudioPredictor
from recorder import record_audio
from voice_auth import registrar_usuario, verificar_voz
from similar_finder import clasificar_silaba_por_vecinos

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Voice Classifier AI - ESCOM",
    page_icon="🎙️",
    layout="wide"
)

# --- CARGA DE MODELOS (CACHED) ---
@st.cache_resource
def load_predictor():
    return AudioPredictor()

predictor = load_predictor()

# --- ESTADO DE LA SESIÓN (STATE MANAGMENT) ---
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
if "usuario" not in st.session_state:
    st.session_state["usuario"] = None
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None
if "seccion_actual" not in st.session_state:
    # Secciones posibles: "Identificación de Sílabas", "Registrar Usuario", "Validar Usuario (Login)"
    st.session_state["seccion_actual"] = "Identificación de Sílabas"

# --- FUNCIONES DE AUDIO ---
def normalizar_audio(x):
    x = np.asarray(x, dtype=float)
    if x.ndim > 1: 
        x = np.mean(x, axis=1)
    max_val = np.max(np.abs(x)) + 1e-12
    return x / max_val

def plot_wave(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    t = np.linspace(0, len(y)/sr, len(y))
    ax.plot(t, y, color='#00FFCC', linewidth=0.7)  # Estilo de alta visibilidad
    ax.set_facecolor('#111625')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.1, color='white')
    st.pyplot(fig)


# --- BARRA LATERAL (SIDEBAR NAVIGATION) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00FFCC;'>🎙️ Control de Voz</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 12px;'>Ingeniería en IA - ESCOM</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Mostrar estado de autenticación actual
    if st.session_state["autenticado"]:
        st.success(f"🔓 Sesión: {st.session_state['usuario']}")
        if st.button("🔒 Cerrar Sesión", use_container_width=True, type="secondary"):
            st.session_state["autenticado"] = False
            st.session_state["usuario"] = None
            st.session_state["seccion_actual"] = "Identificación de Sílabas"
            st.rerun()
    else:
        st.warning("👤 Modo: Invitado / Sin cuenta")

    st.markdown("### 🗺️ Navegación")
    
    # Botón 1: Clasificador base (Acceso libre siempre)
    if st.button("🔍 Analizador de Sílabas & Género", use_container_width=True):
        st.session_state["seccion_actual"] = "Identificación de Sílabas"
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🔐 Seguridad Bio-Vocal")
    
    # Botones solicitados para gestión de usuarios
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("👤 Registrar", use_container_width=True, help="Registrar una nueva huella vocal"):
            st.session_state["seccion_actual"] = "Registrar Usuario"
            st.rerun()
    with col_btn2:
        if st.button("🔑 Validar", use_container_width=True, type="primary" if not st.session_state["autenticado"] else "secondary", help="Iniciar sesión por voz"):
            st.session_state["seccion_actual"] = "Validar Usuario (Login)"
            st.rerun()

    st.markdown("---")
    # Herramientas de administración rápidas
    with st.expander("🛠️ Utilidades del Sistema"):
        if st.button("🔄 Forzar Sincronización KNN", use_container_width=True):
            from similar_finder import crear_base_similitud
            try:
                crear_base_similitud()
                st.toast("Base de similitud actualizada con éxito.", icon="✅")
            except Exception as e:
                st.error(f"Error: {e}")


# --- RENDERIZADO DEL PANEL PRINCIPAL SEGÚN SECCIÓN ---

# 1. SECCIÓN: IDENTIFICACIÓN (ANALIZADOR WEB PRINCIPAL)
if st.session_state["seccion_actual"] == "Identificación de Sílabas":
    st.title("🎙️ Analizador Espectral y Clasificador de Voz")
    st.markdown("Esta sección utiliza el dataset de entrenamiento base y KNN para identificar características.")
    
    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.markdown("### 🎛️ Captura en Vivo")
        dur = st.slider("Duración de la grabación (segundos)", 1, 5, 3)
        
        if st.button("🔴 Iniciar Grabación", type="primary", use_container_width=True):
            # ⚡ RESPONSIVIDAD INMEDIATA: Avisamos al usuario en el milisegundo exacto del clic
            st.toast("🎤 ¡Micrófono activo! Habla ahora...", icon="🔴")
            aviso_grabando = st.error("🎙️ GRABANDO AUDIO... POR FAVOR HABLA")
            
            os.makedirs("data/raw", exist_ok=True)
            temp_path = f"data/raw/capture_{time.time_ns()}.wav"
            
            # Ejecuta la captura física (sd.rec)
            path = record_audio(temp_path, duration=dur)
            
            # ⚡ Elminamos el mensaje de "Terminado". Borramos el aviso de grabación al instante
            aviso_grabando.empty()
            
            if path and os.path.exists(path):
                # Procesamiento silencioso en segundo plano
                sr, y = wavfile.read(path)
                y = normalizar_audio(y)
                features = extract_features(y, sr)
                
                # Inferencia de los modelos
                gen = predictor.predict_genero(features[0])
                sil, similares = clasificar_silaba_por_vecinos(
                    features, csv_base="data/processed/base_similitud.csv"
                )
                
                # Guardamos en sesión para renderizar la interfaz
                st.session_state["resultado"] = {
                    "genero": gen, "silaba": sil, "y": y, "sr": sr, 
                    "features": features, "path": path, "similares": similares
                }
                st.toast("Análisis espectral completado", icon="📊")
            else:
                st.error("No se pudo detectar entrada de audio válida.")

        # Despliegue inmediato del reproductor y la onda si existen resultados
        if st.session_state["resultado"]:
            st.markdown("#### Reproducción del Audio Capturado")
            st.audio(st.session_state["resultado"]["path"])
            plot_wave(st.session_state["resultado"]["y"], st.session_state["resultado"]["sr"])

    with col_results:
        st.markdown("### 📊 Predicciones del Sistema")
        if st.session_state["resultado"]:
            res = st.session_state["resultado"]
            
            m1, m2 = st.columns(2)
            m1.metric(label="F0 - Género Estimado", value=res["genero"])
            m2.metric(label="KNN - Sílaba Detectada", value=res["silaba"])

            with st.expander("🔍 Desglose de Características Espectrales extraídas", expanded=True):
                f = res["features"]
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**F0 (Frecuencia Fundamental):** {f[0]:.2f} Hz")
                    st.write(f"**Formante 1 (F1):** {f[1]:.2f} Hz")
                    st.write(f"**Formante 2 (F2):** {f[2]:.2f} Hz")
                    st.write(f"**Frecuencia Dominante (Fdom):** {f[5]:.2f} Hz")
                with c2:
                    st.write(f"**Centroide Espectral:** {f[6]:.2f}")
                    st.write(f"**Energía de Banda Baja:** {f[4]:.6f}")
                    st.write(f"**Relación de Energías (E):** {f[8]:.6f}")
        else:
            st.info("El panel está listo. Presiona el botón rojo para capturar tu voz.")

# 2. SECCIÓN: REGISTRAR NUEVO USUARIO
elif st.session_state["seccion_actual"] == "Registrar Usuario":
    st.title("👤 Panel de Registro Biométrico Vocal")
    st.markdown("Graba una huella vocal única para que el sistema aprenda a reconocerte de forma personalizada.")
    
    nombre = st.text_input("Ingresa el identificador o nombre del usuario:", placeholder="Ej. Juan_Perez")
    
    st.info("Al presionar el botón, el sistema activará el micrófono de inmediato para procesar tus muestras de voz.")
    
    if st.button("🎤 Comenzar Grabación de Registro", type="primary", use_container_width=True):
        if nombre.strip():
            with st.spinner(f"Grabando muestras para '{nombre.strip()}'... Mantén el micrófono despejado."):
                # Llama a tu función local de voice_auth.py
                exito, mensaje = registrar_usuario(nombre.strip())
                if exito:
                    st.success(f"✨ {mensaje}")
                    st.balloons()
                    # Redirigir automáticamente a la validación para probar
                    st.session_state["seccion_actual"] = "Validar Usuario (Login)"
                else:
                    st.error(f"Hubo un error en el registro: {mensaje}")
        else:
            st.warning("⚠️ Debes rellenar el campo del nombre antes de proceder.")

# 3. SECCIÓN: VALIDAR USUARIO (LOGIN DE DOS FASES)
elif st.session_state["seccion_actual"] == "Validar Usuario (Login)":
    st.title("🔐 Validación de Identidad Dinámica de Dos Fases")
    
    if "fase_auth" not in st.session_state:
        st.session_state["fase_auth"] = 1
    if "f0_calibrado" not in st.session_state:
        st.session_state["f0_calibrado"] = None

    if st.session_state["autenticado"]:
        st.success(f"🎉 Autenticado correctamente como: **{st.session_state['usuario']}**")
        if st.button("🔄 Probar con otro usuario", use_container_width=True):
            st.session_state["autenticado"] = False
            st.session_state["usuario"] = None
            st.session_state["fase_auth"] = 1
            st.session_state["f0_calibrado"] = None
            st.rerun()
            
    else:
        # --- FASE 1: CALIBRACIÓN DE FRECUENCIA ---
        if st.session_state["fase_auth"] == 1:
            st.subheader("Fase 1: Calibración del Tono de Voz ($F_0$)")
            st.info("Por favor, lee el siguiente texto en voz alta para analizar tu frecuencia fundamental:")
            
            st.markdown(
                "> *\"La ingeniería en inteligencia artificial en la Escuela Superior de Cómputo "
                "desarrolla sistemas capaces de procesar señales biométricas complejas.\"*"
            )
            
            dur_calib = st.slider("Duración de lectura (segundos)", 3, 6, 4, key="dur_c")
            
            if st.button("🔴 Iniciar Lectura de Calibración", type="primary", use_container_width=True):
                st.toast("🎤 Analizando tono fundamental... Lee el texto", icon="🔴")
                aviso = st.error("🎙️ ESCUCHANDO TEXTO DE CALIBRACIÓN...")
                
                os.makedirs("data/auth", exist_ok=True)
                path_c = "data/auth/calibracion.wav"
                
                # Graba el texto largo
                record_audio(path_c, duration=dur_calib)
                aviso.empty()
                
                if os.path.exists(path_c):
                    sr, y = wavfile.read(path_c)
                    y = normalizar_audio(y)
                    features = extract_features(y, sr)
                    
                    # Extraemos y guardamos F0 (Frecuencia Fundamental)
                    st.session_state["f0_calibrado"] = features[0]
                    st.session_state["fase_auth"] = 2  # Avanzamos de fase
                    st.toast("Tono calibrado con éxito", icon="✅")
                    st.rerun()
                else:
                    st.error("No se pudo procesar la calibración.")

        # --- FASE 2: DESAFÍO DE LA FRASE ---
        elif st.session_state["fase_auth"] == 2:
            st.subheader("Fase 2: Verificación de Frase de Acceso")
            st.warning(f"Tono base detectado: {st.session_state['f0_calibrado']:.2f} Hz. Ahora di tu frase rápida.")
            
            st.markdown("🗣️ **Frase a decir:** *\"Acceso Seguro Protocolo Alfa\"*")
            
            if st.button("🎙️ Grabar Frase de Verificación", type="primary", use_container_width=True):
                st.toast("🎤 Escuchando frase...", icon="🔴")
                aviso = st.error("🎙️ DI LA FRASE AHORA...")
                
                login_path = "data/auth/login_current.wav"
                record_audio(login_path, duration=3)
                aviso.empty()
                
                if os.path.exists(login_path):
                    # 1. Validación de identidad estándar contra tu BD
                    acceso, usuario, distancia, mensaje = verificar_voz(login_path, umbral=180.0)
                    
                    # 2. Validación cruzada de F0 (Comparamos la calibración vs la frase corta)
                    sr, y = wavfile.read(login_path)
                    y = normalizar_audio(y)
                    features_frase = extract_features(y, sr)
                    f0_frase = features_frase[0]
                    
                    # Margen de tolerancia aceptable en Hz entre el texto largo y la frase corta (ej. 25 Hz)
                    tolerancia_f0 = 25.0
                    diferencia_f0 = abs(st.session_state["f0_calibrado"] - f0_frase)
                    
                    if acceso and (diferencia_f0 <= tolerancia_f0):
                        st.session_state["autenticado"] = True
                        st.session_state["usuario"] = usuario
                        st.success(f"🎉 ¡Acceso Autorizado! Bienvenido {usuario}")
                        st.balloons()
                        time.sleep(1.5)
                        st.session_state["seccion_actual"] = "Identificación de Sílabas"
                        st.rerun()
                    else:
                        st.session_state["fase_auth"] = 1  # Reiniciar si falla
                        if diferencia_f0 > tolerancia_f0:
                            st.error(f"❌ Acceso Denegado: Desfase de tono detectado ({diferencia_f0:.2f} Hz de diferencia). ¡Posible suplantación por grabación!")
                        else:
                            st.error(f"❌ Acceso Denegado. {mensaje}")
                else:
                    st.error("No se pudo capturar la frase.")