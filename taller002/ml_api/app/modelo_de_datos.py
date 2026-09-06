
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ..., min_length=1, description="Vector de features de entrada"
    )


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: Optional[str] = None
    model_file: str
    model_trained_at: Optional[str] = None
