import time
import numpy as np

import transcripcion
import traduccion


class Pipeline:

    def __init__(self):
        self.whisper = transcripcion.cargar_modelo()
        self.traductores = traduccion.cargar_modelos()
        self.idioma_previo = None
        self._calentar()

    def _calentar(self):
        # La primera pasada por la GPU siempre es lenta; la hacemos al arrancar
        silencio = np.zeros(transcripcion.SR_WHISPER, dtype=np.float32)
        transcripcion.detectar_idioma(self.whisper, silencio)
        for origen in ("es", "en"):
            traduccion.traducir(self.traductores, "hola", origen)

    def procesar(self, audio_16k):
        """Recibe audio float32 mono a 16 kHz. Regresa un dict con resultados y tiempos (ms)."""
        t0 = time.perf_counter()
        idioma, confianza = transcripcion.detectar_idioma(self.whisper, audio_16k, self.idioma_previo)
        texto = transcripcion.transcribir(self.whisper, audio_16k, idioma)
        t1 = time.perf_counter()
        traducido, destino = traduccion.traducir(self.traductores, texto, idioma)
        t2 = time.perf_counter()

        if texto:
            self.idioma_previo = idioma

        return {
            "idioma": idioma,
            "confianza_idioma": confianza,
            "destino": destino,
            "texto": texto,
            "traduccion": traducido,
            "ms_transcripcion": (t1 - t0) * 1000,
            "ms_traduccion": (t2 - t1) * 1000,
            "ms_total": (t2 - t0) * 1000,
            "duracion_audio_s": len(audio_16k) / transcripcion.SR_WHISPER,
        }
