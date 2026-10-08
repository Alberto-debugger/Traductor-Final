import torch
from transformers import MarianMTModel, MarianTokenizer


MODELOS_MT = {
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
}
DESTINO = {"es": "en", "en": "es"}


def cargar_modelos(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelos = {}
    for par, nombre in MODELOS_MT.items():
        tokenizer = MarianTokenizer.from_pretrained(nombre)
        modelo = MarianMTModel.from_pretrained(nombre).to(device).eval()
        modelos[par] = (tokenizer, modelo)
    print(f"Traductores cargados en {device}")
    return modelos


def traducir(modelos, texto, origen):
    """Traduce `texto` del idioma `origen` al otro. Regresa (traducción, idioma_destino)."""
    destino = DESTINO[origen]
    if not texto:
        return "", destino

    tokenizer, modelo = modelos[(origen, destino)]
    entrada = tokenizer([texto], return_tensors="pt", padding=True).to(modelo.device)
    with torch.inference_mode():
        salida = modelo.generate(**entrada, num_beams=4)
    return tokenizer.decode(salida[0], skip_special_tokens=True), destino
