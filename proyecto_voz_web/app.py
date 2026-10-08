import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import numpy as np
import warnings

# --- TUS MÓDULOS ---
import ia_texto
import experto_genero
import filtros_digitales  # <--- NUEVO IMPORT

warnings.filterwarnings("ignore")

# Cargar IA
try:
    cerebro_ia = ia_texto.cargar_modelo()
except:
    cerebro_ia = None

# --- GRÁFICA: 64 MUESTRAS (ESTILO ECUALIZADOR) ---
def crear_grafica_64_muestras(audio_path, y_limpio, sr):
    if audio_path is None: return None
    
    # 1. Preparar datos de tiempo (Onda)
    # Usamos el audio YA FILTRADO (y_limpio)
    # Limitamos a 4000 puntos para que la gráfica de onda no sea pesada
    muestras_onda = 4000
    if len(y_limpio) > muestras_onda:
        paso = len(y_limpio) // muestras_onda
        y_viz = y_limpio[::paso]
        tiempo = np.linspace(0, len(y_limpio) / sr, num=len(y_viz))
    else:
        y_viz = y_limpio
        tiempo = np.linspace(0, len(y_limpio) / sr, num=len(y_limpio))

    # 2. Transformada para las 64 BARRAS
    # Usamos Mel-Spectrogram para agrupar la FFT en 64 bandas perceptuales
    n_bandas = 64
    mel_spec = librosa.feature.melspectrogram(
        y=y_limpio, sr=sr, 
        n_mels=n_bandas, 
        fmax=8000 # Solo nos interesa hasta 8kHz (voz humana)
    )
    
    # Convertimos a decibelios (Volumen)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Promediamos en el tiempo para obtener "La huella" estática del audio
    # Esto nos da un array de exactamente 64 valores
    barras_64 = np.mean(mel_db, axis=1)
    
    # Normalizamos visualmente para que se vea bonito en la gráfica (-80dB a 0dB)
    barras_64 = np.clip(barras_64, -80, 0) + 80 # Lo pasamos a positivo (0 a 80 altura)
    
    # Crear eje X para las 64 barras (simulado en Hz)
    frecuencias_mel = librosa.mel_frequencies(n_mels=n_bandas, fmax=8000)
    eje_x_barras = [f"{int(f)} Hz" for f in frecuencias_mel]

    # --- SUBPLOTS ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=("🌊 SEÑAL FILTRADA (Digital Butterworth)", f"📊 ESPECTRO FFT ({n_bandas} MUESTRAS)")
    )

    # Gráfica 1: Onda (Línea)
    fig.add_trace(go.Scatter(
        x=tiempo, y=y_viz, mode='lines', 
        line=dict(color='#00F3FF', width=1), 
        name='Amplitud'
    ), row=1, col=1)

    # Gráfica 2: 64 Barras (Bar Chart)
    fig.add_trace(go.Bar(
        x=eje_x_barras, 
        y=barras_64,
        marker=dict(
            color=barras_64,
            colorscale='Viridis', # Escala de colores profesional
            showscale=False
        ),
        name='Energía'
    ), row=2, col=1)

    # Layout Profesional
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif", size=10, color="#ccc"),
        margin=dict(l=40, r=20, t=40, b=40), 
        height=500,
        showlegend=False,
        modebar=dict(bgcolor='rgba(255, 255, 255, 0.1)'),
        bargap=0.1 # Espacio entre las barras
    )
    
    return fig

# --- PROCESAMIENTO ---
def procesar_dashboard(audio_path):
    if audio_path is None: return "Sin audio", None, None, None, None

    # A. Cargar audio crudo
    y, sr = librosa.load(audio_path, duration=4.0)

    # B. APLICAR FILTROS DIGITALES (La Magia ✨)
    y_limpio = filtros_digitales.procesar_senal(y, sr)
    
    # Guardamos temporalmente el audio limpio por si Whisper lo prefiere
    # (Opcional, pero Whisper suele preferir audio raw, así que usaremos el limpio solo para analisis y grafica)
    
    # 1. TEXTO (Whisper)
    texto = ia_texto.transcribir_audio(cerebro_ia, audio_path) if cerebro_ia else "Error IA"
    
    # 2. GÉNERO (Usamos el audio original o limpio? Probemos con el original para no alterar F0)
    genero, pitch, probabilidad, color_hex = experto_genero.analizar_voz_completo(audio_path)

    # 3. GRÁFICA (Con 64 Muestras y Audio Filtrado)
    fig = crear_grafica_64_muestras(audio_path, y_limpio, sr)

    # 4. CARDS HTML
    html_principal = f"""
    <div class="result-box" style="border-left: 5px solid {color_hex};">
        <div class="label-mini" style="color:{color_hex}">GÉNERO DETECTADO</div>
        <div class="value-main">{genero}</div>
        <div class="sub-value" style="color: #aaa;">Probabilidad: {probabilidad}</div>
    </div>
    """
    
    html_detalles = f"""
    <div class="result-box" style="border-left: 5px solid #00F3FF;">
        <div class="label-mini" style="color:#00F3FF">F0 (PITCH)</div>
        <div class="value-main">{pitch:.0f} Hz</div>
        <div class="sub-value" style="color: #aaa;">Frecuencia Fundamental</div>
    </div>
    """

    return texto, html_principal, html_detalles, fig

# --- ESTILOS CSS ---
css_dashboard = """
body { background-color: #050505; color: white; font-family: 'Segoe UI', sans-serif; }
.gradio-container { max-width: 1200px !important; margin: auto; }
.result-box {
    background: rgba(25, 25, 25, 0.8); border-radius: 8px; padding: 20px; height: 100%;
    display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.label-mini { font-size: 0.8rem; font-weight: bold; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase; }
.value-main { font-size: 2.5rem; font-weight: 800; margin-bottom: 5px; color: white; }
.sub-value { font-size: 0.9rem; }
#titulo-app {
    text-align: center; background: linear-gradient(90deg, #00C6FF, #0072FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 3rem; font-weight: 900; margin-bottom: 20px;
}
"""

tema = gr.themes.Base(primary_hue="cyan", neutral_hue="neutral").set(
    body_background_fill="#050505", block_background_fill="#111", block_border_color="#333",
    input_background_fill="#151515"
)

# --- LAYOUT ---
with gr.Blocks(theme=tema, css=css_dashboard, title="Voice Lab Pro + Filtros") as demo:
    
    gr.Markdown("# 🎙️ VOICE LAB: DSP + IA", elem_id="titulo-app")

    # SECCIÓN 1
    with gr.Row():
        with gr.Column(scale=3):
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", show_label=False)
        with gr.Column(scale=1):
            btn = gr.Button("🔍 ANALIZAR CON FILTROS", variant="primary", size="lg")
            gr.Markdown("<div style='text-align:center; color:#555; margin-top:10px'>Filtro: Butterworth Bandpass</div>")

    # SECCIÓN 2
    gr.Markdown("### 🧬 Análisis de Voz")
    with gr.Row():
        with gr.Column(scale=1):
            out_card_main = gr.HTML()
            out_card_detail = gr.HTML()
        with gr.Column(scale=1):
            out_text = gr.TextArea(label="Transcripción", lines=8, interactive=False)

    # SECCIÓN 3
    gr.Markdown("### 🎚️ Visualización Digital (64 Bandas)")
    with gr.Row():
        out_plot = gr.Plot(show_label=False, container=False)

    btn.click(
        fn=procesar_dashboard,
        inputs=input_audio,
        outputs=[out_text, out_card_main, out_card_detail, out_plot]
    )

if __name__ == "__main__":
    demo.launch(share=True)