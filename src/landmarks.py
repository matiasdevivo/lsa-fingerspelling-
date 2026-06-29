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


N_HAND_FEATURES = N_LANDMARKS * 3  # 63, por mano
N_FACE_KEYPOINTS = 6
N_FACE_FEATURES = N_FACE_KEYPOINTS * 2  # 12 (x, y por punto clave)
N_FRAME_FEATURES_DINAMICO = 2 * N_HAND_FEATURES + N_FACE_FEATURES  # 138


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


def extract_hands_landmarks(frame, hands_detector, max_hands=2):
    """
    Extrae landmarks de hasta `max_hands` manos detectadas en el frame.
    A diferencia de extract_landmarks, soporta más de una mano y conserva
    la etiqueta Left/Right de cada una.

    Args:
        frame: imagen BGR (np.array) capturada con OpenCV.
        hands_detector: instancia de mp.solutions.hands.Hands ya creada
            (debe tener max_num_hands >= max_hands para detectar todas).
        max_hands: cantidad máxima de manos a devolver.

    Returns:
        Lista de dicts [{"label": "Left"/"Right", "landmarks": np.array(63,)}, ...]
        en coordenadas crudas (sin normalizar). Lista vacía si no se
        detectó ninguna mano.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands_detector.process(frame_rgb)

    manos = []
    if not resultado.multi_hand_landmarks:
        return manos

    for mano, handedness in zip(
        resultado.multi_hand_landmarks, resultado.multi_handedness
    ):
        coords = []
        for punto in mano.landmark:
            coords.extend([punto.x, punto.y, punto.z])
        manos.append({
            "label": handedness.classification[0].label,  # "Left" o "Right"
            "landmarks": np.array(coords, dtype=np.float32),
        })

    return manos[:max_hands]


def extract_face_landmarks(frame, face_detector):
    """
    Detecta el rostro y devuelve sus puntos clave (ojos, nariz, boca, orejas),
    útil para letras que se hacen cerca de o en contacto con la cara.

    Args:
        frame: imagen BGR (np.array) capturada con OpenCV.
        face_detector: instancia de mp.solutions.face_detection.FaceDetection
            ya creada.

    Returns:
        np.array (12,) con 6 puntos clave (x, y) en coordenadas relativas
        de la imagen (0-1), o None si no se detectó rostro.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_detector.process(frame_rgb)

    if not resultado.detections:
        return None

    deteccion = resultado.detections[0]
    coords = []
    for punto in deteccion.location_data.relative_keypoints:
        coords.extend([punto.x, punto.y])

    return np.array(coords, dtype=np.float32)


def hands_to_vector(manos_detectadas):
    """
    Convierte la lista de hasta 2 manos detectadas (extract_hands_landmarks)
    en un vector fijo de 126 features. Las manos faltantes se completan
    con ceros, para que toda muestra tenga siempre la misma forma.
    """
    vector = np.zeros(2 * N_HAND_FEATURES, dtype=np.float32)
    for i, mano in enumerate(manos_detectadas[:2]):
        vector[i * N_HAND_FEATURES:(i + 1) * N_HAND_FEATURES] = mano["landmarks"]
    return vector


def face_to_vector(landmarks_cara):
    """Devuelve el vector de cara (12,), o ceros si no se detectó rostro."""
    if landmarks_cara is None:
        return np.zeros(N_FACE_FEATURES, dtype=np.float32)
    return landmarks_cara


def remuestrear_secuencia(secuencia, frames_objetivo=20):
    """
    Remuestrea una secuencia (T, n_features) a una longitud fija de
    frames mediante interpolación lineal, para que todas las muestras
    de una seña dinámica tengan la misma forma sin importar cuántos
    frames se hayan capturado originalmente.
    """
    n_frames_original, n_features = secuencia.shape
    indices_originales = np.linspace(0, n_frames_original - 1, n_frames_original)
    indices_objetivo = np.linspace(0, n_frames_original - 1, frames_objetivo)

    secuencia_remuestreada = np.empty((frames_objetivo, n_features), dtype=np.float32)
    for col in range(n_features):
        secuencia_remuestreada[:, col] = np.interp(
            indices_objetivo, indices_originales, secuencia[:, col]
        )
    return secuencia_remuestreada
