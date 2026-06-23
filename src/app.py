"""
Interfaz Gradio para el reconocimiento del alfabeto dactilológico de LSA.

Toma una imagen de la webcam (vía Gradio), extrae y normaliza los
landmarks de la mano con MediaPipe, y predice la letra usando el
clasificador entrenado en models/modelo.joblib.
"""

import os
import sys

import gradio as gr
import joblib
import mediapipe as mp

sys.path.append(os.path.dirname(__file__))
from landmarks import extract_landmarks, normalize_landmarks

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo.joblib")

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6,
)

modelo = joblib.load(MODEL_PATH) if os.path.isfile(MODEL_PATH) else None


def predecir_letra(imagen):
    """Recibe una imagen RGB (np.array) desde Gradio y devuelve la letra predicha."""
    if modelo is None:
        return "Modelo no encontrado. Corré src/train.py primero."

    if imagen is None:
        return "Esperando imagen..."

    # Gradio entrega RGB, OpenCV/MediaPipe esperan BGR para extract_landmarks
    frame_bgr = imagen[:, :, ::-1]

    landmarks_crudos = extract_landmarks(frame_bgr, hands_detector)
    if landmarks_crudos is None:
        return "Sin mano detectada"

    landmarks_normalizados = normalize_landmarks(landmarks_crudos)
    prediccion = modelo.predict([landmarks_normalizados])[0]
    return prediccion


with gr.Blocks(title="Reconocimiento de Alfabeto Dactilológico LSA") as demo:
    gr.Markdown(
        "# Reconocimiento de Alfabeto Dactilológico LSA\n"
        "Mostrá una letra del alfabeto dactilológico de LSA frente a la cámara."
    )
    entrada = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
    salida = gr.Textbox(label="Letra detectada")

    # stream_every limita la frecuencia de frames procesados (uno cada 0.5s
    # en vez de a la velocidad máxima de la cámara), para evitar que en
    # conexiones remotas (como Hugging Face Spaces) se acumulen frames y
    # el video se vea entrecortado/vibrando.
    entrada.stream(
        fn=predecir_letra,
        inputs=entrada,
        outputs=salida,
        stream_every=0.5,
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
