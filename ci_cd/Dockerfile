# Un solo Dockerfile, tres etapas (multi-stage):
#   base    -> dependencias de runtime (compartidas, se cachean)
#   test    -> base + pytest; su CMD ejecuta las pruebas   (Actividad 1)
#   runtime -> base + solo el codigo de la API; es la imagen que se publica (Actividad 2)
# docker build --target test    -t readmission-api:test .
# docker build --target runtime -t readmission-api:dev  .

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Primero requirements y despues el codigo: si solo cambia el codigo,
# Docker reutiliza la capa del pip install (builds mucho mas rapidos).
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM base AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY . .
# El codigo de salida de pytest (0 = todo paso) es el codigo de salida del contenedor.
CMD ["pytest", "-q"]

FROM base AS runtime
COPY app/ app/
RUN useradd --create-home --uid 10001 appuser
USER appuser
ENV MODEL_PATH=/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
