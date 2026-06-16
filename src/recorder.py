import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np

DEVICE_ID = 1

def record_audio(filename="data/raw/temp_recording.wav", duration=3):
    try:
        print(f"🎙️ Grabando por {duration} segundos...")

        # Info del dispositivo
        device_info = sd.query_devices(DEVICE_ID, 'input')
        fs = int(device_info['default_samplerate'])

        print("Usando dispositivo:", device_info['name'])
        print("Sample rate:", fs)


        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            device=DEVICE_ID
        )
        sd.wait()

        # Verificar señal
        max_val = np.max(np.abs(recording))
        print(f"🔊 Nivel máximo: {max_val}")

        if max_val < 0.01:
            print("⚠️ Señal muy baja (posible silencio)")

        # Crear carpeta
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Convertir a int16
        recording_int16 = np.int16(recording * 32767)

        # Guardar
        write(filename, fs, recording_int16)

        print(f"✅ Guardado en: {filename}")
        return filename

    except Exception as e:
        print("❌ Error al grabar audio:", e)
        return None