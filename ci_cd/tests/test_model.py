"""Pruebas de app/model.py. Las rapidas usan el modelo falso; la lenta entrena uno."""

import joblib
import pytest

from app.model import ModelNotAvailableError, load_model, predict_readmission
from app.preprocessing import FEATURE_ORDER, build_features


def test_predict_readmission_returns_probability_and_risk(fake_model, valid_record):
    result = predict_readmission(fake_model, build_features(valid_record))
    assert result == {"readmission_probability": 0.75, "risk": "high"}


def test_predict_readmission_sends_columns_in_model_order(fake_model, valid_record):
    """El modelo debe recibir un DataFrame de 1 fila con las columnas en orden."""
    predict_readmission(fake_model, build_features(valid_record))
    frame = fake_model.last_input
    assert list(frame.columns) == FEATURE_ORDER
    assert len(frame) == 1


def test_load_model_missing_file_raises(tmp_path):
    # tmp_path: fixture de pytest con una carpeta temporal unica por prueba
    with pytest.raises(ModelNotAvailableError, match="no hay modelo"):
        load_model(str(tmp_path / "no_existe.joblib"))


def test_load_model_reads_joblib_artifact(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"tipo": "artefacto de prueba"}, path)
    assert load_model(str(path)) == {"tipo": "artefacto de prueba"}


@pytest.mark.slow
def test_demo_model_learns_the_synthetic_rule():
    """Integracion preprocesamiento + sklearn: entrenar y exigir un minimo.
    Es lenta (entrena), por eso esta marcada: en cada push corre -m 'not slow'."""
    from sklearn.metrics import accuracy_score

    from train_demo_model import synthetic_records, train

    model = train(n=1500, seed=42)
    X_test, y_test = synthetic_records(500, seed=7)  # otra semilla = datos no vistos
    assert accuracy_score(y_test, model.predict(X_test)) >= 0.90
