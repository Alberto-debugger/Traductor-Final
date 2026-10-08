import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel


MODELO_WHISPER = "large-v3-turbo"
SR_WHISPER = 16000          
IDIOMAS = ("es", "en")      
UMBRAL_IDIOMA = 0.6         


def cargar_modelo(nombre=MODELO_WHISPER):
    
    try:
        modelo = WhisperModel(nombre, device="cuda", compute_type="float16")
        print(f"Whisper '{nombre}' cargado en GPU")
    except Exception as e:
        print(f"No se pudo usar la GPU ({e}); cargando en CPU")
        modelo = WhisperModel(nombre, device="cpu", compute_type="int8")
    return modelo


def preparar_audio(sr, datos):
    
    datos = np.asarray(datos)
    if np.issubdtype(datos.dtype, np.integer):
        datos = datos.astype(np.float32) / np.iinfo(datos.dtype).max
    datos = datos.astype(np.float32)
    if datos.ndim == 2:
        datos = datos.mean(axis=1)
    if sr != SR_WHISPER:
        datos = resample_poly(datos, SR_WHISPER, sr).astype(np.float32)
    return datos


def detectar_idioma(modelo, audio, idioma_previo=None):
    
    _, _, probs = modelo.detect_language(audio)
    probs = dict(probs)
    p_es, p_en = probs.get("es", 0.0), probs.get("en", 0.0)
    total = p_es + p_en
    if total == 0:
        return idioma_previo or "es", 0.0

    p_es, p_en = p_es / total, p_en / total
    idioma, confianza = ("es", p_es) if p_es >= p_en else ("en", p_en)

    if confianza < UMBRAL_IDIOMA and idioma_previo in IDIOMAS:
        return idioma_previo, confianza
    return idioma, confianza


def transcribir(modelo, audio, idioma):
    
    segmentos, _ = modelo.transcribe(
        audio,
        language=idioma,
        beam_size=5,
        vad_filter=True,                    
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segmentos).strip()
