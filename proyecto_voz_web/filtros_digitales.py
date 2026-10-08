
import numpy as np
import scipy.signal as signal
import librosa

def aplicar_filtro_butterworth(y, sr, corte_bajo=80, corte_alto=8000, orden=4):
    """
    Aplica un filtro Pasa-Banda (Bandpass) Digital.
    - Elimina ruido < 80Hz (golpes, viento).
    - Elimina ruido > 8000Hz (silbidos electrónicos).
    - Usa un filtro Butterworth de orden 4 (estándar en audio).
    """
    try:
        # Frecuencia de Nyquist (mitad de la tasa de muestreo)
        nyquist = 0.5 * sr
        
        # Normalizar las frecuencias de corte (0 a 1)
        bajo = corte_bajo / nyquist
        alto = corte_alto / nyquist
        
        # Crear los coeficientes del filtro (b, a)
        b, a = signal.butter(orden, [bajo, alto], btype='band')
        
        # Aplicar el filtro a la señal
        y_filtrada = signal.lfilter(b, a, y)
        
        return y_filtrada
    except Exception as e:
        print(f"Error en filtro digital: {e}")
        return y # Si falla, devolvemos el audio original

def normalizar_audio(y):
    """
    Normaliza el volumen para que siempre esté al nivel óptimo (-1.0 a 1.0)
    """
    if np.max(np.abs(y)) == 0:
        return y
    return y / np.max(np.abs(y))

def procesar_senal(y, sr):
    """
    Pipeline completo de limpieza.
    """
    # 1. Filtro de Frecuencias
    y = aplicar_filtro_butterworth(y, sr)
    
    # 2. Normalización de Amplitud
    y = normalizar_audio(y)
    
    # 3. Pre-énfasis (opcional): Resalta frecuencias altas para mejor inteligibilidad de voz
    # y = librosa.effects.preemphasis(y) 
    
    return y