import whisper
import warnings

warnings.filterwarnings("ignore")

def cargar_modelo():
    print("⏳ Cargando modelo Whisper (Base)...")
    # Usamos 'base' para equilibrio entre velocidad y precisión
    model = whisper.load_model("base")
    print("✅ Modelo Whisper listo.")
    return model

def transcribir_audio(model, audio_path):
    """
    Transcribe el audio forzando español.
    """
    try:
        if audio_path is None:
            return ""
            
        # fp16=False es necesario para CPUs (tu i7)
        # language="es" fuerza al modelo a reconocer español
        result = model.transcribe(
            audio_path, 
            fp16=False, 
            language="es", 
            temperature=0.0
        )
        
        texto = result["text"].strip()
        
        if not texto:
            return "(No se detectaron palabras claras)"
            
        return texto

    except Exception as e:
        return f"Error en transcripción: {str(e)}"