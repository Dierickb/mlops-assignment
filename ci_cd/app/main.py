"""API FastAPI: solo traduce HTTP <-> funciones. La logica vive en otros modulos."""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import ModelNotAvailableError, load_model, predict_readmission
from app.preprocessing import build_features

app = FastAPI(title="Diabetes Readmission API", version="0.1.0")


class PatientRecord(BaseModel):
    """Contrato de entrada. Pydantic valida TIPOS y campos obligatorios (-> 422).
    Los RANGOS los valida app.preprocessing (-> 400): asi la regla de negocio
    vive en un solo lugar y se reutiliza fuera de la API."""

    age: str = Field(examples=["[70-80)"])
    time_in_hospital: float = Field(examples=[3])
    num_lab_procedures: float = Field(examples=[41])
    num_procedures: float = Field(examples=[0])
    num_medications: float = Field(examples=[15])
    number_outpatient: float = Field(examples=[0])
    number_emergency: float = Field(examples=[0])
    number_inpatient: float = Field(examples=[1])
    number_diagnoses: float = Field(examples=[9])
    race: str | None = Field(default=None, examples=["Caucasian"])
    gender: str | None = Field(default=None, examples=["Female"])
    A1Cresult: str | None = Field(default=None, examples=[">8"])
    insulin: str | None = Field(default=None, examples=["Steady"])
    change: str | None = Field(default=None, examples=["Ch"])
    diabetesMed: str | None = Field(default=None, examples=["Yes"])


@lru_cache(maxsize=1)
def _cached_model():
    # lru_cache: el modelo se lee del disco UNA vez, no en cada peticion.
    return load_model()


def get_model():
    """Dependencia de FastAPI. En las pruebas se reemplaza con
    app.dependency_overrides[get_model] = lambda: modelo_falso."""
    try:
        return _cached_model()
    except ModelNotAvailableError as exc:
        # 503 = el servicio esta vivo pero aun no puede atender (falta el modelo)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    """Liveness: responde aunque no haya modelo (no depende de get_model)."""
    return {"status": "ok"}


@app.post("/predict")
def predict(record: PatientRecord, model=Depends(get_model)):
    try:
        features = build_features(record.model_dump())
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return predict_readmission(model, features)
