# Traductor-Final

| Carpeta | Qué hace |
|---|---|
| `traductor/` | Hablas en español o inglés y aparece el texto traducido al otro idioma (Whisper large-v3-turbo + MarianMT) |
| `proyecto_voz_web/` | Analiza una grabación: la transcribe con Whisper, estima el género por tono y timbre, y grafica la señal filtrada y su espectro en 64 bandas |

## Antes de empezar (una sola vez)

- Python 3.12
- Solo para `proyecto_voz_web`: ffmpeg. Se instala con `winget install ffmpeg` y luego se abre una terminal nueva.

## Instalación

Cada carpeta lleva su propio entorno. Desde `traductor/` o desde `proyecto_voz_web/`:

```
python -m venv venv
venv\Scripts\activate
```

PyTorch, según la computadora:

```
# Con tarjeta NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Sin tarjeta NVIDIA (funciona, pero Whisper va más lento)
pip install torch
```

Luego el resto:

```
pip install -r requirements.txt
```

## Uso

En `traductor/`:

```
python probar.py      # prueba sin micrófono con los audios de pruebas/
python app.py         # interfaz en http://127.0.0.1:7860
```

En `proyecto_voz_web/`:

```
python app.py         # interfaz en http://127.0.0.1:7860 (también crea un enlace público temporal)
```

La primera vez se descargan los modelos: unos 2.2 GB para `traductor` y unos 140 MB para `proyecto_voz_web`.

Para cerrar cualquiera de las dos interfaces: Ctrl + C en la terminal.
