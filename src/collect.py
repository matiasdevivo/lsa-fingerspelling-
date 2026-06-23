"""
Script de captura de dataset para el alfabeto dactilológico de LSA.

Uso:
    python src/collect.py --letter A --samples 80

Controles:
    SPACE -> capturar una muestra (si hay mano detectada)
    Q     -> guardar y salir antes de completar las muestras
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp

# Permite ejecutar el script tanto como módulo (python -m src.collect)
# como directamente (python src/collect.py) sin romper el import.
sys.path.append(os.path.dirname(__file__))
from landmarks import extract_landmarks, normalize_landmarks, N_LANDMARKS

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def _construir_header():
    """Construye el encabezado del CSV: l0x, l0y, l0z, ..., l20z, label."""
    columnas = []
    for i in range(N_LANDMARKS):
        columnas.extend([f"l{i}x", f"l{i}y", f"l{i}z"])
    columnas.append("label")
    return columnas


def _guardar_muestras(letter, muestras):
    """Guarda (o agrega) las muestras capturadas en data/raw/{LETTER}.csv."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    ruta_csv = os.path.join(RAW_DATA_DIR, f"{letter}.csv")

    existe = os.path.isfile(ruta_csv)

    # Modo append: si el archivo ya existe, se agregan filas sin reescribir el header
    with open(ruta_csv, mode="a", newline="") as archivo:
        escritor = csv.writer(archivo)
        if not existe:
            escritor.writerow(_construir_header())
        for muestra in muestras:
            fila = list(muestra) + [letter]
            escritor.writerow(fila)

    print(f"Guardadas {len(muestras)} muestras nuevas en {ruta_csv}")


def capturar_dataset(letter, n_samples):
    """Abre la webcam y permite capturar n_samples landmarks normalizados para 'letter'."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    muestras = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands_detector:

        ventana = f"LSA Capture - Letra: {letter}"

        while cap.isOpened() and len(muestras) < n_samples:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la webcam.")
                break

            frame = cv2.flip(frame, 1)  # efecto espejo, más natural para el usuario

            landmarks_crudos = extract_landmarks(frame, hands_detector)

            # Dibujamos los landmarks sobre el frame si se detectó una mano
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands_detector.process(frame_rgb)
            if resultado.multi_hand_landmarks:
                for mano in resultado.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

            # Texto de progreso
            texto_progreso = f"{letter}: {len(muestras)}/{n_samples}"
            cv2.putText(
                frame, texto_progreso, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            if landmarks_crudos is None:
                cv2.putText(
                    frame, "Sin mano detectada", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            cv2.imshow(ventana, frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break

            if tecla == ord(" "):
                if landmarks_crudos is None:
                    print("Sin mano detectada, no se capturó la muestra.")
                else:
                    muestra_normalizada = normalize_landmarks(landmarks_crudos)
                    muestras.append(muestra_normalizada)
                    print(f"Muestra capturada: {len(muestras)}/{n_samples}")

    cap.release()
    cv2.destroyAllWindows()

    if muestras:
        _guardar_muestras(letter, muestras)
    else:
        print("No se capturó ninguna muestra, no se guarda nada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura un dataset de landmarks normalizados para una letra del alfabeto LSA."
    )
    parser.add_argument("--letter", required=True, help="Letra a capturar, ej: A")
    parser.add_argument("--samples", type=int, default=80, help="Cantidad de muestras a capturar")
    args = parser.parse_args()

    capturar_dataset(args.letter.upper(), args.samples)
