"""
Extracción y normalización de landmarks de mano (MediaPipe Hands)
para el reconocimiento del alfabeto dactilológico de LSA.
"""

import cv2
import numpy as np

N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3  # x, y, z por landmark


def extract_landmarks(frame, hands_detector):
    """
    Extrae los 21 landmarks de la mano detectada en un frame BGR de OpenCV.

    Args:
        frame: imagen BGR (np.array) capturada con OpenCV.
        hands_detector: instancia de mp.solutions.hands.Hands ya creada.

    Returns:
        np.array de shape (63,) con [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
        o None si no se detectó ninguna mano en el frame.
    """
    # MediaPipe espera imágenes en formato RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands_detector.process(frame_rgb)

    if not resultado.multi_hand_landmarks:
        return None

    # Tomamos la primera mano detectada
    mano = resultado.multi_hand_landmarks[0]

    coords = []
    for punto in mano.landmark:
        coords.extend([punto.x, punto.y, punto.z])

    return np.array(coords, dtype=np.float32)


def normalize_landmarks(landmarks_array):
    """
    Normaliza un array de landmarks (63,) para hacerlo invariante
    a la posición y a la distancia de la mano respecto a la cámara.

    Pasos:
        1. Se traslada el origen al landmark 0 (muñeca).
        2. Se escala por la distancia máxima de cualquier landmark
           al landmark 0, dejando los valores aproximadamente en [-1, 1].

    Args:
        landmarks_array: np.array (63,) con landmarks crudos de MediaPipe.

    Returns:
        np.array (63,) normalizado.
    """
    puntos = landmarks_array.reshape(N_LANDMARKS, 3).copy()

    # 1. Origen en la muñeca (landmark 0)
    origen = puntos[0].copy()
    puntos -= origen

    # 2. Escala por la distancia máxima al origen (ya trasladado, es el (0,0,0))
    distancias = np.linalg.norm(puntos, axis=1)
    escala = distancias.max()

    # Evitar división por cero en el caso degenerado de mano sin tamaño
    if escala == 0:
        escala = 1e-6

    puntos /= escala

    return puntos.reshape(-1).astype(np.float32)
