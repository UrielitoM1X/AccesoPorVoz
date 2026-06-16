import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np

DEVICE_ID = 1

def record_audio(filename="data/raw/temp_recording.wav", max_duration=10, silence_threshold=0.015):
    """
    Graba audio de forma dinámica. Se detiene automáticamente si detecta 
    silencio prolongado tras haber identificado voz.
    """
    try:
        device_info = sd.query_devices(DEVICE_ID, 'input')
        fs = int(device_info['default_samplerate'])
        
        chunk_duration = 0.2  # Procesar bloques de 200ms
        chunk_samples = int(fs * chunk_duration)
        
        recording_chunks = []
        
        has_spoken = False
        silence_counter = 0
        max_chunks = int(max_duration / chunk_duration)
        
        # Iniciar el flujo de entrada
        with sd.InputStream(samplerate=fs, channels=1, device=DEVICE_ID, dtype='float32') as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                recording_chunks.append(chunk)
                
                # Calcular el nivel de energía del bloque actual (RMS)
                max_val = np.max(np.abs(chunk))
                
                if max_val > silence_threshold:
                    if not has_spoken:
                        has_spoken = True  # El usuario empezó a hablar
                    silence_counter = 0
                else:
                    if has_spoken:
                        silence_counter += 1
                
                # Si ya habló y lleva ~1.4 segundos en silencio, se corta automáticamente
                if has_spoken and silence_counter >= 7:
                    break
                    
        if not recording_chunks:
            return None
            
        recording = np.concatenate(recording_chunks, axis=0)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        recording_int16 = np.int16(recording * 32767)
        write(filename, fs, recording_int16)
        
        return filename

    except Exception as e:
        print("❌ Error en grabación dinámica:", e)
        return None