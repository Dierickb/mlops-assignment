from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str


from pydantic import BaseModel


class PenguinFeatures(BaseModel):
    unnamed: int
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: int
    year: int
    island_Dream: int
    island_Torgersen: int
    sex_male: int

@app.post("/items/")
def create_item(item: Item):
    return {"message": "Item creado", "item": item}