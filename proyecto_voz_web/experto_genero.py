import librosa
import numpy as np

def analizar_voz_completo(audio_path):
    """
    Algoritmo de Decisión Binaria Estricta (Votación Ponderada).
    Calcula un Score de 0 a 100.
    - Score < 50: HOMBRE
    - Score >= 50: MUJER
    """
    try:
        # 1. Cargar audio (4 segundos)
        y, sr = librosa.load(audio_path, duration=4.0)
        
        # --- ANÁLISIS 1: PITCH (F0) ---
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('D4'))
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) == 0:
            return "Indeterminado", 0, "0%"

        pitch_mediana = np.median(f0)

        # --- ANÁLISIS 2: TIMBRE (Centroide Espectral) ---
        centroide = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroide_medio = np.mean(centroide)
        
        # --- SISTEMA DE PUNTUACIÓN (0 = Muy Hombre, 100 = Muy Mujer) ---
        # Empezamos en el centro (50/50)
        score = 50.0 
        
        # A) Votos por Tono (Pitch) - Ajuste más agresivo
        if pitch_mediana < 120: score -= 35
        elif pitch_mediana < 155: score -= 20
        elif pitch_mediana < 172: score -= 10 # Zona baja del traslape
        
        if pitch_mediana > 230: score += 35
        elif pitch_mediana > 190: score += 20
        elif pitch_mediana >= 172: score += 10 # Zona alta del traslape

        # B) Votos por Timbre (Brillo)
        # Ayuda a decidir en la zona de 160-180Hz
        if centroide_medio < 1600: score -= 15
        elif centroide_medio > 2400: score += 15
        
        # Limitar el score entre 0 y 100 para que sea un porcentaje válido
        score = max(0, min(100, score))

        # --- DECISIÓN BINARIA ESTRICTA ---
        genero = ""
        probabilidad_texto = ""

        if score >= 50:
            genero = "MUJER 👩"
            probabilidad = score # Ej: 60%
            color = "#FF4081" # Rosa
        else:
            genero = "HOMBRE 👨"
            probabilidad = 100 - score # Si score es 20, probabilidad hombre es 80%
            color = "#00E676" # Verde

        probabilidad_texto = f"{probabilidad:.1f}% de Probabilidad"

        return genero, pitch_mediana, probabilidad_texto, color

    except Exception as e:
        return "Error", 0, str(e), "#888"