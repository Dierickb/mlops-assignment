"""Carga del modelo y prediccion.

La API nunca llama a joblib directamente: pasa por estas funciones.
Asi, en las pruebas, se reemplaza el modelo por uno falso sin tocar
el disco, MLflow ni MinIO.
"""

import os
from pathlib import Path

import joblib
import pandas as pd

from app.preprocessing import FEATURE_ORDER, probability_to_risk

# La ruta llega por variable de entorno: el mismo codigo sirve en local,
# en Docker (-v ./models:/models) y en Kubernetes (volumen o initContainer).
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.joblib")


class ModelNotAvailableError(RuntimeError):
    """El artefacto del modelo no existe en MODEL_PATH."""


def load_model(path: str = MODEL_PATH):
    """Carga un modelo serializado con joblib (p. ej. un Pipeline de sklearn)."""
    if not Path(path).is_file():
        raise ModelNotAvailableError(f"no hay modelo en {path}")
    return joblib.load(path)


def predict_readmission(model, features: dict) -> dict:
    """Probabilidad de reingreso y etiqueta de riesgo para UN registro.

    `model` solo necesita un metodo predict_proba(DataFrame) -> [[p0, p1], ...]
    (contrato de sklearn). La columna 1 es la clase positiva: reingreso.
    """
    frame = pd.DataFrame([features], columns=FEATURE_ORDER)
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "readmission_probability": round(probability, 4),
        "risk": probability_to_risk(probability),
    }
