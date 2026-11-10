"""Fixtures compartidos. pytest los descubre solo: no hay que importarlos."""

import pytest
from fastapi.testclient import TestClient

from app.main import _cached_model, app, get_model


@pytest.fixture
def valid_record() -> dict:
    """Un registro valido, tal como llegaria en el JSON de /predict.
    Scope function (por defecto): cada prueba recibe una copia nueva,
    asi una prueba que borra una clave no contamina a las demas."""
    return {
        "age": "[70-80)",
        "time_in_hospital": 3,
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 9,
        "race": "Caucasian",
        "gender": "Female",
        "A1Cresult": ">8",
        "insulin": "Steady",
        "change": "Ch",
        "diabetesMed": "Yes",
    }


class FakeModel:
    """Modelo falso con el mismo contrato que sklearn (predict_proba).
    Siempre responde 0.75 de reingreso y guarda lo que recibio,
    para poder verificar QUE le entrego la API al modelo."""

    def __init__(self, positive_probability: float = 0.75):
        self.positive_probability = positive_probability
        self.last_input = None

    def predict_proba(self, X):
        self.last_input = X
        p = self.positive_probability
        return [[1 - p, p] for _ in range(len(X))]


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def client(fake_model):
    """Cliente HTTP en memoria con el modelo real reemplazado por el falso."""
    app.dependency_overrides[get_model] = lambda: fake_model
    yield TestClient(app)
    app.dependency_overrides.clear()  # limpieza: no afectar otras pruebas


@pytest.fixture
def client_without_model(monkeypatch):
    """Cliente SIN override: simula que el archivo del modelo no existe."""
    from app.model import ModelNotAvailableError

    def _missing():
        raise ModelNotAvailableError("no hay modelo en /models/model.joblib")

    _cached_model.cache_clear()  # el cache podria tener un modelo de otra prueba
    monkeypatch.setattr("app.main.load_model", _missing)
    yield TestClient(app)
    _cached_model.cache_clear()
