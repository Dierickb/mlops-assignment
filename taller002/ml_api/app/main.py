from fastapi import FastAPI
from taller002.ml_api.app.metodos import router as metodos_router
from taller002.ml_api.app.respuestas_y_estados import router as respuestas_router

app = FastAPI(
    title="API de Penguin",
    description="API para predicción de especies de pingüinos",
    version="1.0.0"
)

app.include_router(metodos_router)
app.include_router(respuestas_router)

@app.get("/")
def home():
    return {"message": "¡Hola, FastAPI está funcionando!"}
