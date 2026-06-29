"""
Interfaz Gradio para el reconocimiento del alfabeto dactilológico de LSA.

Tiene dos pestañas:
- Letras estáticas: reconocimiento en vivo con el clasificador Random
  Forest entrenado en models/modelo.joblib (1 mano, una foto = una letra).
- Letras dinámicas: reconocimiento de gestos en movimiento (como CH, LL)
  con la LSTM entrenada en models/modelo_dinamico.keras. Acumula frames
  durante una ventana de tiempo fija y predice sobre la secuencia
  completa, no sobre un frame suelto.

La pestaña dinámica depende de TensorFlow, que no se instala en el
despliegue de Hugging Face Spaces por su peso — pensada para correrse
en local.
"""

import json
import os
import sys
import time

import gradio as gr
import joblib
import mediapipe as mp
import numpy as np

sys.path.append(os.path.dirname(__file__))
from landmarks import (
    extract_landmarks,
    normalize_landmarks,
    extract_hands_landmarks,
    extract_face_landmarks,
    hands_to_vector,
    face_to_vector,
    remuestrear_secuencia,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo.joblib")
MODEL_DINAMICO_PATH = os.path.join(BASE_DIR, "models", "modelo_dinamico.keras")
LABELS_DINAMICO_PATH = os.path.join(BASE_DIR, "models", "letras_dinamicas.json")

DURACION_GRABACION = 2.5
FRAMES_FIJOS = 20

# --- Letras estáticas ---

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


# --- Letras dinámicas (opcional, requiere TensorFlow) ---

try:
    from tensorflow import keras

    modelo_dinamico = (
        keras.models.load_model(MODEL_DINAMICO_PATH)
        if os.path.isfile(MODEL_DINAMICO_PATH)
        else None
    )
    if os.path.isfile(LABELS_DINAMICO_PATH):
        with open(LABELS_DINAMICO_PATH) as archivo:
            letras_dinamicas = json.load(archivo)
    else:
        letras_dinamicas = None
except ImportError:
    modelo_dinamico = None
    letras_dinamicas = None

mp_face = mp.solutions.face_detection
hands_detector_dinamico = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)


def iniciar_grabacion_dinamica(estado):
    """Reinicia el estado para empezar a grabar una nueva seña dinámica."""
    estado = {"grabando": True, "inicio": time.time(), "frames": []}
    return estado, "Grabando..."


def procesar_frame_dinamico(imagen, estado):
    """
    Callback de streaming para la pestaña dinámica. Mientras estado["grabando"]
    es True, acumula el vector de manos+cara de cada frame; al completarse
    la ventana de tiempo, remuestrea la secuencia y predice con la LSTM.

    Devuelve dos valores: el texto de estado/countdown (se actualiza todo
    el tiempo) y la letra detectada (solo se actualiza al terminar una
    grabación; el resto del tiempo se deja sin tocar con gr.skip() para
    que no desaparezca apenas se muestra).
    """
    if modelo_dinamico is None or letras_dinamicas is None:
        return "Modelo dinámico no encontrado. Corré src/train_dynamic.py primero.", gr.skip()

    if imagen is None or not estado.get("grabando"):
        return "Apretá 'Grabar seña' para empezar.", gr.skip()

    frame_bgr = imagen[:, :, ::-1]
    manos_detectadas = extract_hands_landmarks(frame_bgr, hands_detector_dinamico, max_hands=2)
    landmarks_cara = extract_face_landmarks(frame_bgr, face_detector)

    tiempo_transcurrido = time.time() - estado["inicio"]

    if tiempo_transcurrido >= DURACION_GRABACION:
        estado["grabando"] = False
        frames = estado["frames"]

        if len(frames) < 2:
            return "No se capturaron suficientes frames, probá de nuevo.", gr.skip()

        secuencia = np.stack(frames)
        secuencia_remuestreada = remuestrear_secuencia(secuencia, FRAMES_FIJOS)
        entrada_modelo = secuencia_remuestreada[None, ...]  # batch de 1

        probabilidades = modelo_dinamico.predict(entrada_modelo, verbose=0)[0]
        indice_predicho = int(np.argmax(probabilidades))
        letra_predicha = letras_dinamicas[indice_predicho]

        texto_resultado = f"{letra_predicha} (confianza: {probabilidades[indice_predicho]:.0%})"
        return "Listo. Apretá 'Grabar seña' para repetir.", texto_resultado

    vector_manos = hands_to_vector(manos_detectadas)
    vector_cara = face_to_vector(landmarks_cara)
    estado["frames"].append(np.concatenate([vector_manos, vector_cara]))

    restante = DURACION_GRABACION - tiempo_transcurrido
    return f"Grabando... {restante:.1f}s", gr.skip()


with gr.Blocks(title="Reconocimiento de Alfabeto Dactilológico LSA") as demo:
    gr.Markdown("# Reconocimiento de Alfabeto Dactilológico LSA")

    with gr.Tabs():
        with gr.Tab("Letras estáticas"):
            gr.Markdown("Mostrá una letra del alfabeto dactilológico de LSA frente a la cámara.")
            with gr.Row():
                entrada_estatica = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                salida_estatica = gr.Textbox(label="Letra detectada")

            # stream_every limita la frecuencia de frames procesados (uno cada
            # 0.5s en vez de a la velocidad máxima de la cámara), para evitar
            # que en conexiones remotas (como Hugging Face Spaces) se
            # acumulen frames y el video se vea entrecortado/vibrando.
            entrada_estatica.stream(
                fn=predecir_letra,
                inputs=entrada_estatica,
                outputs=salida_estatica,
                stream_every=0.5,
            )

        with gr.Tab("Letras dinámicas"):
            gr.Markdown(
                "Apretá 'Grabar seña' y hacé el gesto durante los "
                f"{DURACION_GRABACION} segundos siguientes (ej: CH, LL)."
            )
            with gr.Row():
                entrada_dinamica = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                with gr.Column():
                    estado_texto = gr.Textbox(
                        label="Estado", value="Apretá 'Grabar seña' para empezar."
                    )
                    letra_detectada_dinamica = gr.Textbox(label="Letra detectada")

            boton_grabar = gr.Button("Grabar seña (2.5s)")
            estado_grabacion = gr.State({"grabando": False, "inicio": None, "frames": []})

            boton_grabar.click(
                fn=iniciar_grabacion_dinamica,
                inputs=estado_grabacion,
                outputs=[estado_grabacion, estado_texto],
            )

            # stream_every más corto que en estáticas: necesitamos varios
            # frames dentro de la ventana de 2.5s para reconstruir el gesto.
            # letra_detectada_dinamica solo se actualiza al terminar una
            # grabación (el resto del tiempo la función devuelve gr.skip()
            # para ese output), así el resultado queda fijo en pantalla.
            entrada_dinamica.stream(
                fn=procesar_frame_dinamico,
                inputs=[entrada_dinamica, estado_grabacion],
                outputs=[estado_texto, letra_detectada_dinamica],
                stream_every=0.1,
            )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
