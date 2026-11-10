"""Pruebas del contrato HTTP con TestClient (en memoria, sin levantar uvicorn)."""

import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_ok(client, valid_record):
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 200
    assert response.json() == {"readmission_probability": 0.75, "risk": "high"}


def test_predict_accepts_missing_tokens_in_categoricals(client, valid_record):
    valid_record["race"] = "?"
    valid_record["gender"] = "Unknown/Invalid"
    assert client.post("/predict", json=valid_record).status_code == 200


@pytest.mark.parametrize("field", ["age", "num_medications"])
def test_predict_missing_required_field_returns_422(client, valid_record, field):
    del valid_record[field]
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 422  # lo rechaza Pydantic (contrato)


def test_predict_wrong_type_returns_422(client, valid_record):
    valid_record["num_medications"] = "muchas"
    assert client.post("/predict", json=valid_record).status_code == 422


def test_predict_out_of_range_returns_400(client, valid_record):
    valid_record["time_in_hospital"] = 40
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 400  # lo rechaza nuestra regla de negocio
    assert "time_in_hospital" in response.json()["detail"]


def test_predict_bad_age_bracket_returns_400(client, valid_record):
    valid_record["age"] = "75"
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 400
    assert "rango de edad" in response.json()["detail"]


def test_predict_without_model_returns_503(client_without_model, valid_record):
    response = client_without_model.post("/predict", json=valid_record)
    assert response.status_code == 503


def test_health_works_without_model(client_without_model):
    """Liveness no depende del modelo: Kubernetes no debe reiniciar el pod por eso."""
    assert client_without_model.get("/health").status_code == 200
