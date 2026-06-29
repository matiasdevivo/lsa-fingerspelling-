"""
Entrenamiento del clasificador de señas dinámicas de LSA (gestos en
movimiento, como CH y LL), a partir de las secuencias capturadas con
collect_dynamic.py en data/sequences/{LETRA}/*.csv.

A diferencia de train.py (Random Forest para letras estáticas), este
script entrena una red LSTM, porque el problema acá es clasificar una
secuencia de frames en el tiempo, no un único vector de landmarks.

Uso:
    python src/train_dynamic.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

sys.path.append(os.path.dirname(__file__))
from landmarks import remuestrear_secuencia

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SEQUENCES_DIR = os.path.join(BASE_DIR, "data", "sequences")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_dinamico.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "letras_dinamicas.json")

FRAMES_FIJOS = 20  # longitud fija a la que se remuestrea cada secuencia


def cargar_dataset():
    """Carga todas las secuencias de data/sequences/, remuestreadas a longitud fija."""
    if not os.path.isdir(SEQUENCES_DIR):
        raise FileNotFoundError(
            f"No existe {SEQUENCES_DIR}. Corré src/collect_dynamic.py primero."
        )

    letras = sorted(
        nombre for nombre in os.listdir(SEQUENCES_DIR)
        if os.path.isdir(os.path.join(SEQUENCES_DIR, nombre))
    )
    if not letras:
        raise FileNotFoundError(
            f"No hay subcarpetas de letras en {SEQUENCES_DIR}. "
            "Corré src/collect_dynamic.py primero."
        )

    secuencias = []
    etiquetas = []

    for letra in letras:
        carpeta_letra = os.path.join(SEQUENCES_DIR, letra)
        archivos = sorted(
            f for f in os.listdir(carpeta_letra) if f.endswith(".csv")
        )
        for archivo in archivos:
            df = pd.read_csv(os.path.join(carpeta_letra, archivo))
            df = df.drop(columns=["frame_idx"])
            secuencia = df.to_numpy(dtype=np.float32)
            secuencias.append(remuestrear_secuencia(secuencia, FRAMES_FIJOS))
            etiquetas.append(letra)

    X = np.stack(secuencias)
    y = np.array(etiquetas)
    return X, y


def construir_modelo(n_features, n_clases):
    """Arma una red LSTM simple para clasificar secuencias de landmarks."""
    modelo = keras.Sequential([
        keras.layers.Input(shape=(FRAMES_FIJOS, n_features)),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_clases, activation="softmax"),
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def entrenar():
    X, y = cargar_dataset()
    print(f"Dataset cargado: {X.shape[0]} secuencias, {X.shape[1]} frames, {X.shape[2]} features")

    codificador = LabelEncoder()
    y_codificado = codificador.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codificado, test_size=0.2, random_state=42, stratify=y_codificado
    )

    modelo = construir_modelo(n_features=X.shape[2], n_clases=len(codificador.classes_))
    modelo.summary()

    modelo.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=8,
        verbose=2,
    )

    y_pred = np.argmax(modelo.predict(X_test), axis=1)
    print(classification_report(
        y_test, y_pred, target_names=codificador.classes_, labels=range(len(codificador.classes_))
    ))

    os.makedirs(MODELS_DIR, exist_ok=True)
    modelo.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as archivo:
        json.dump(list(codificador.classes_), archivo)

    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"Letras guardadas en {LABELS_PATH}")


if __name__ == "__main__":
    entrenar()
