from fastapi import APIRouter
import pandas as pd
from taller001.modelo_de_datos import PenguinFeatures
from taller001.carga_modelo import model

router = APIRouter()

@router.post("/predict")
def predict_species(data: PenguinFeatures):

    input_data = pd.DataFrame([{
        "Unnamed: 0": data.unnamed,
        "bill_length_mm": data.bill_length_mm,
        "bill_depth_mm": data.bill_depth_mm,
        "flipper_length_mm": data.flipper_length_mm,
        "body_mass_g": data.body_mass_g,
        "year": data.year,
        "island_Dream": data.island_Dream,
        "island_Torgersen": data.island_Torgersen,
        "sex_male": data.sex_male
    }])

    prediction = model.predict(input_data)

    return {
        "predicted_species": prediction[0]
    }