# LSA Fingerspelling

Reconocimiento del alfabeto dactilológico de la Lengua de Señas Argentina (LSA)
usando MediaPipe Hands + scikit-learn, desplegado con Gradio en Hugging Face Spaces.

🔗 Demo en vivo: https://huggingface.co/spaces/matiascodeds/lsa-fingerspelling

Proyecto académico: a partir de los 21 puntos de la mano que detecta
MediaPipe, normalizados y convertidos en un vector de 63 features, un
clasificador de scikit-learn predice qué letra está haciendo el usuario.
No se procesan imágenes crudas con redes neuronales, solo coordenadas
numéricas de landmarks.

## Estructura

- `data/raw/` — CSV con landmarks normalizados por letra (letras estáticas).
- `data/sequences/{LETRA}/` — secuencias de frames para señas dinámicas.
- `models/` — modelos entrenados (`modelo.joblib`, `modelo_dinamico.keras`).
- `src/landmarks.py` — extracción y normalización de landmarks.
- `src/collect.py` — captura interactiva de dataset estático con webcam.
- `src/collect_dynamic.py` — captura de secuencias para señas dinámicas.
- `src/train.py` — entrenamiento del clasificador estático (Random Forest).
- `src/train_dynamic.py` — entrenamiento del clasificador dinámico (LSTM).
- `src/app.py` — interfaz Gradio (punto de entrada del Space).

## Uso local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow==2.15.0  # solo para señas dinámicas

# Capturar muestras de una letra estática (SPACE = guardar, Q = salir)
python src/collect.py --letter A --samples 80

# Entrenar el modelo estático con todo lo capturado
python src/train.py

# Capturar secuencias de una seña dinámica (SPACE = iniciar grabación de 2.5s, Q = salir)
python src/collect_dynamic.py --letter LL --samples 30

# Entrenar el modelo dinámico (LSTM) con las secuencias capturadas
python src/train_dynamic.py

# Correr la app (pestaña dinámica requiere TensorFlow instalado)
python src/app.py
```
