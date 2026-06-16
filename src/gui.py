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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Control de acceso por voz",
    page_icon="🔐",
    layout="centered"
)

# --- ESTADO DE LA SESIÓN ---
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
if "usuario" not in st.session_state:
    st.session_state["usuario"] = None
if "fase_reg" not in st.session_state:
    st.session_state["fase_reg"] = 1
if "nombre_registro" not in st.session_state:
    st.session_state["nombre_registro"] = ""
if "seccion_actual" not in st.session_state:
    st.session_state["seccion_actual"] = "Validar Acceso"

def normalizar_audio(x):
    x = np.asarray(x, dtype=float)
    if x.ndim > 1: x = np.mean(x, axis=1)
    return x / (np.max(np.abs(x)) + 1e-12)

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00FFCC;'>Seguridad Vocal</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state["autenticado"]:
        st.success(f"Bienvenido: {st.session_state['usuario']}")
        if st.button("Cerrar Sesión", use_container_width=True):
            st.session_state["autenticado"] = False
            st.session_state["usuario"] = None
            st.session_state["seccion_actual"] = "Validar Acceso"
            st.rerun()
    else:
        st.warning("Estado: Bloqueado")

    st.markdown("### Acciones")
    if st.button("Validar Acceso", use_container_width=True, type="primary" if st.session_state["seccion_actual"] == "Validar Acceso" else "secondary"):
        st.session_state["seccion_actual"] = "Validar Acceso"
        st.rerun()
        
    if st.button("Registrar Nuevo Usuario", use_container_width=True, type="primary" if st.session_state["seccion_actual"] == "Registrar Usuario" else "secondary"):
        st.session_state["seccion_actual"] = "Registrar Usuario"
        st.rerun()

# --- PANEL PRINCIPAL ---

# SECCIÓN 1: LOGIN / VALIDACIÓN DE ACCESO
if st.session_state["seccion_actual"] == "Validar Acceso":
    st.title("Control de Acceso mediante Reconocimiento de Voz")
    
    if st.session_state["autenticado"]:
        st.balloons()
        st.success(f"Sistema desbloqueado correctamente. Hola de nuevo, {st.session_state['usuario']}.")
    else:
        st.markdown("🗣️ **Frase a decir:** *\"Autenticación mediante mi firma de voz digital.\"*")
        
        if st.button("🎙️ Iniciar Reconocimiento de Voz", use_container_width=True, type="primary"):
            st.toast("🎤 Micrófono abierto... Di tu frase de acceso", icon="🔴")
            aviso = st.error("🎙️ ESCUCHANDO... (Habla ahora y el sistema se detendrá al terminar de leer)")
            
            os.makedirs("data/auth", exist_ok=True)
            login_path = "data/auth/login_current.wav"
            
            # Llama a la grabación dinámica sin tiempos fijos
            path = record_audio(login_path)
            aviso.empty()
            
            if path and os.path.exists(path):
                st.toast("Audio capturado. Procesando analíticas...", icon="⚡")
                acceso, usuario, distancia, mensaje = verificar_voz(path, umbral=0.12)
                
                if acceso:
                    st.session_state["autenticado"] = True
                    st.session_state["usuario"] = usuario
                    st.success(f"Acceso Concedido. Reconocido como: {usuario} (Distancia: {distancia:.2f})")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Acceso Denegado. {mensaje}")
                    st.warning(f"💡 Distancia Coseno obtenida: {distancia:.4f} (Se requiere que sea menor o igual a 0.12)")
            else:
                st.error("No se detectó señal o el entorno tiene demasiado ruido de fondo.")

# SECCIÓN 2: REGISTRO DE USUARIO POR PASOS DINÁMICOS
elif st.session_state["seccion_actual"] == "Registrar Usuario":
    st.title("👤 Registro de un Nuevo Usuario")
    st.markdown("Completa los pasos para completar el registro.")

    # PASO 1: Ingreso de Datos
    if st.session_state["fase_reg"] == 1:
        st.markdown("### Paso 1: Identificación del Sujeto")
        nombre_input = st.text_input("Nombre único de registro:", placeholder="Ej. Alumno_ESCOM")
        
        if st.button("Siguiente Paso", use_container_width=True, type="primary"):
            if nombre_input.strip():
                st.session_state["nombre_registro"] = nombre_input.strip()
                st.session_state["fase_reg"] = 2
                st.rerun()
            else:
                st.warning("Por favor, ingresa un nombre para continuar.")

    # PASO 2: Grabación Dinámica sin Reloj
    elif st.session_state["fase_reg"] == 2:
        st.markdown(f"### Paso 2: Grabación para **{st.session_state['nombre_registro']}**")
        st.info("Presiona el botón y lee de corrido el texto de calibración. El sistema detectará automáticamente cuando termines.")
        
        st.markdown(
    "> *\"El viejo circo de madera bamboleaba su enorme carpa azul sobre el suelo arenoso, "
    "mientras la música resonaba con fuerza y el público aplaudía entusiasmado cada truco.\"*"
)
        
        if st.button("Empezar a Leer", use_container_width=True, type="primary"):
            st.toast("🎤 Grabadora lista... Empieza a leer el texto", icon="🔴")
            aviso = st.error("🎙️ LEYENDO... (El sistema se cerrará automáticamente al detectar tu silencio final)")
            
            # Guardado temporal en data/raw para procesamiento de firmas
            os.makedirs("data/raw", exist_ok=True)
            reg_path = f"data/raw/reg_{st.session_state['nombre_registro']}.wav"
            
            path = record_audio(reg_path)
            aviso.empty()
            
            if path and os.path.exists(path):
                st.session_state["path_registro_temp"] = path
                st.session_state["fase_reg"] = 3
                st.toast("Lectura procesada con éxito", icon="✅")
                st.rerun()
            else:
                st.error("Error al procesar la lectura. Intenta hablar con mayor claridad.")

    # PASO 3: Validación y Almacenamiento Final
    elif st.session_state["fase_reg"] == 3:
        st.markdown("### Paso 3: Validación de Parámetros")
        st.success("Audio capturado exitosamente de forma responsiva.")
        
        # Muestra el reproductor para que el usuario verifique lo que grabó
        st.audio(st.session_state["path_registro_temp"])
        
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            if st.button("Volver a Grabar Muestra", use_container_width=True):
                st.session_state["fase_reg"] = 2
                st.rerun()
                
        with col_reg2:
            if st.button("Guardar y Validar en Sistema", use_container_width=True, type="primary"):
                with st.spinner("Compilando vectores de frecuencia en la Base de Datos..."):
                    # Vincula la muestra real capturada con tu script interno de voice_auth
                    exito, mensaje = registrar_usuario(st.session_state["nombre_registro"])
                    
                    if exito:
                        st.success(f"Huella vocal de '{st.session_state['nombre_registro']}' guardada en el sistema.")
                        st.toast("Registro Exitoso", icon="🎉")
                        # Resetear estados de flujo y mandar al login
                        st.session_state["fase_reg"] = 1
                        st.session_state["seccion_actual"] = "Validar Acceso"
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"Error al escribir vectores: {mensaje}")