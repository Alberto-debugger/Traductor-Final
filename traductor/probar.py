import sys
from pathlib import Path

import soundfile as sf

import transcripcion
from pipeline import Pipeline

carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "pruebas"
pipe = Pipeline()

for archivo in sorted(carpeta.glob("*.wav")):
    datos, sr = sf.read(archivo)
    r = pipe.procesar(transcripcion.preparar_audio(sr, datos))
    print(f"\n{archivo.name}  ({r['duracion_audio_s']:.1f} s de audio)")
    print(f"  Idioma:     {r['idioma']} ({r['confianza_idioma']:.0%})")
    print(f"  Texto:      {r['texto']}")
    print(f"  Traducción: {r['traduccion']}")
    print(f"  Tiempos:    Whisper {r['ms_transcripcion']:.0f} ms + traducción {r['ms_traduccion']:.0f} ms"
          f" = {r['ms_total']:.0f} ms")
