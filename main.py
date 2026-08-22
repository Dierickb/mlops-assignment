from fastapi import FastAPI

from metodos import router as metodos_router
from respuestas_y_estados import router as respuestas_router

app = FastAPI(
    title="Penguin API",
    description="API para prediccion de especies de pinguinos",
    version="1.0.0"
)

app.include_router(metodos_router)
app.include_router(respuestas_router)


@app.get("/")
def home():
    return {
        "message": "¡Hola, FastAPI esta funcionando!"
    }