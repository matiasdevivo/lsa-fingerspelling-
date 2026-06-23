"""
Entrenamiento del clasificador de letras del alfabeto dactilológico de LSA.

Lee todos los CSV de data/raw/*.csv (uno por letra, con landmarks ya
normalizados) y entrena un clasificador de scikit-learn, guardando el
modelo resultante en models/modelo.joblib.

Uso:
    python src/train.py
"""

import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo.joblib")


def cargar_dataset():
    """Carga y concatena todos los CSV de data/raw en un único DataFrame."""
    archivos_csv = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not archivos_csv:
        raise FileNotFoundError(
            f"No se encontraron CSV en {RAW_DATA_DIR}. "
            "Corré src/collect.py primero para generar el dataset."
        )

    dataframes = [pd.read_csv(archivo) for archivo in archivos_csv]
    return pd.concat(dataframes, ignore_index=True)


def entrenar():
    df = cargar_dataset()

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    entrenar()
