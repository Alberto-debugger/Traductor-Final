import gradio as gr

import transcripcion
from pipeline import Pipeline

NOMBRE_IDIOMA = {"es": "Español", "en": "Inglés"}

pipe = Pipeline()


def procesar(audio):
    if audio is None:
        return "", "", ""
    sr, datos = audio
    r = pipe.procesar(transcripcion.preparar_audio(sr, datos))

    info = (
        f"**{NOMBRE_IDIOMA[r['idioma']]} → {NOMBRE_IDIOMA[r['destino']]}** "
        f"(certeza del idioma: {r['confianza_idioma']:.0%})  \n"
        f"Audio: {r['duracion_audio_s']:.1f} s · Whisper: {r['ms_transcripcion']:.0f} ms · "
        f"Traducción: {r['ms_traduccion']:.0f} ms · **Total: {r['ms_total']:.0f} ms**"
    )
    texto = r["texto"] or "(No se detectaron palabras)"
    return info, texto, r["traduccion"]


with gr.Blocks(title="Traductor ") as demo:
    gr.Markdown("Traductor")

    entrada = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Tu voz")
    info = gr.Markdown()
    with gr.Row():
        original = gr.Textbox(label="Lo que dijiste", lines=4, interactive=False)
        traducido = gr.Textbox(label="Traducción", lines=4, interactive=False)

    salidas = [info, original, traducido]
    entrada.stop_recording(procesar, inputs=entrada, outputs=salidas)
    entrada.upload(procesar, inputs=entrada, outputs=salidas)

if __name__ == "__main__":
    demo.launch()
