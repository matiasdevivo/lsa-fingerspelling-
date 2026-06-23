---
title: LSA Fingerspelling
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: '3.11'
app_file: src/app.py
pinned: false
short_description: Reconocimiento del alfabeto LSA con MediaPipe
---

# LSA Fingerspelling

Reconocimiento del alfabeto dactilológico de la Lengua de Señas Argentina (LSA)
usando MediaPipe Hands + scikit-learn, desplegado con Gradio en Hugging Face Spaces.

Es un proyecto académico: a partir de los 21 puntos de la mano que detecta
MediaPipe, normalizados y convertidos en un vector de 63 features, un
clasificador de scikit-learn predice qué letra está haciendo el usuario.
No se procesan imágenes crudas con redes neuronales, solo coordenadas
numéricas de landmarks.

## Estructura

- `data/raw/` — CSV con landmarks normalizados por letra (dataset de entrenamiento).
- `models/` — modelo entrenado (`modelo.joblib`).
- `src/landmarks.py` — extracción y normalización de landmarks.
- `src/collect.py` — captura interactiva de dataset con webcam.
- `src/train.py` — entrenamiento del clasificador.
- `src/app.py` — interfaz Gradio (punto de entrada del Space).

## Uso local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Capturar muestras de una letra
python src/collect.py --letter A --samples 80

# Entrenar el modelo con todo lo capturado
python src/train.py

# Correr la app
python src/app.py
```
