from fastapi import FastAPI

app = FastAPI(title="Proyecto 1 - API de Inferencia (base)")


@app.get("/health")
def health():
    return {"status": "ok"}
