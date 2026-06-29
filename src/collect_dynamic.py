"""
Script de captura de dataset para señas dinámicas de LSA (gestos en
movimiento, como CH y LL), a diferencia de collect.py que captura letras
estáticas con una sola foto.

Cada muestra es una secuencia de frames grabada durante una ventana fija
de tiempo (2.5 segundos), capturando en cada frame los landmarks de hasta
2 manos y del rostro (en coordenadas crudas, sin normalizar, para no
perder la posición relativa mano-cara).

Uso:
    python src/collect_dynamic.py --letter CH --samples 20

Controles:
    SPACE -> iniciar la grabación de una secuencia (2.5 segundos)
    Q     -> salir antes de completar las muestras
"""

import argparse
import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

sys.path.append(os.path.dirname(__file__))
from landmarks import (
    extract_hands_landmarks,
    extract_face_landmarks,
    hands_to_vector,
    face_to_vector,
)

SEQUENCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sequences"
)
DURACION_SEGUNDOS = 2.5


def _construir_header():
    """frame_idx, mano1 (63), mano2 (63), cara (12)."""
    columnas = ["frame_idx"]
    for prefijo in ("h1", "h2"):
        for i in range(21):
            columnas.extend([f"{prefijo}_l{i}x", f"{prefijo}_l{i}y", f"{prefijo}_l{i}z"])
    for i in range(6):
        columnas.extend([f"face_k{i}x", f"face_k{i}y"])
    return columnas


def _guardar_secuencia(letter, frames):
    """Guarda una secuencia completa como un nuevo CSV en data/sequences/{LETTER}/{n}.csv."""
    carpeta_letra = os.path.join(SEQUENCES_DIR, letter)
    os.makedirs(carpeta_letra, exist_ok=True)

    existentes = [f for f in os.listdir(carpeta_letra) if f.endswith(".csv")]
    siguiente_id = len(existentes)
    ruta_csv = os.path.join(carpeta_letra, f"{siguiente_id}.csv")

    with open(ruta_csv, mode="w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(_construir_header())
        for frame_idx, (vector_manos, vector_cara) in enumerate(frames):
            fila = [frame_idx] + list(vector_manos) + list(vector_cara)
            escritor.writerow(fila)

    print(f"Secuencia guardada en {ruta_csv} ({len(frames)} frames)")


def capturar_dataset_dinamico(letter, n_samples):
    """Abre la webcam y permite grabar n_samples secuencias de 'letter'."""
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    secuencias_grabadas = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands_detector, mp_face.FaceDetection(
        min_detection_confidence=0.6
    ) as face_detector:

        ventana = f"LSA Capture Dinámico - Letra: {letter}"
        grabando = False
        inicio_grabacion = None
        frames_secuencia = []

        while cap.isOpened() and secuencias_grabadas < n_samples:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la webcam.")
                break

            frame = cv2.flip(frame, 1)

            manos_detectadas = extract_hands_landmarks(frame, hands_detector, max_hands=2)
            landmarks_cara = extract_face_landmarks(frame, face_detector)

            # Dibujamos los landmarks de las manos detectadas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado_manos = hands_detector.process(frame_rgb)
            if resultado_manos.multi_hand_landmarks:
                for mano in resultado_manos.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

            texto_progreso = f"{letter}: {secuencias_grabadas}/{n_samples}"
            cv2.putText(
                frame, texto_progreso, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            if grabando:
                tiempo_transcurrido = time.time() - inicio_grabacion
                if tiempo_transcurrido >= DURACION_SEGUNDOS:
                    grabando = False
                    if frames_secuencia:
                        _guardar_secuencia(letter, frames_secuencia)
                        secuencias_grabadas += 1
                    frames_secuencia = []
                else:
                    vector_manos = hands_to_vector(manos_detectadas)
                    vector_cara = face_to_vector(landmarks_cara)
                    frames_secuencia.append((vector_manos, vector_cara))

                    restante = DURACION_SEGUNDOS - tiempo_transcurrido
                    cv2.putText(
                        frame, f"GRABANDO {restante:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    )
            elif not manos_detectadas:
                cv2.putText(
                    frame, "Sin mano detectada", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            cv2.imshow(ventana, frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break

            if tecla == ord(" ") and not grabando:
                grabando = True
                inicio_grabacion = time.time()
                frames_secuencia = []

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total de secuencias grabadas en esta sesión: {secuencias_grabadas}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura un dataset de secuencias de landmarks para una seña dinámica de LSA."
    )
    parser.add_argument("--letter", required=True, help="Seña a capturar, ej: CH")
    parser.add_argument("--samples", type=int, default=20, help="Cantidad de secuencias a capturar")
    args = parser.parse_args()

    capturar_dataset_dinamico(args.letter.upper(), args.samples)
