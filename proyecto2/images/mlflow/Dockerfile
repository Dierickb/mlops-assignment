# Servidor de MLflow. Los argumentos de arranque (backend store, bucket, allowed-hosts)
# los define k8s/mlflow/deployment.yaml, no la imagen: la misma imagen sirve en cualquier entorno.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /usr/local/bin/

# MLFLOW_DISABLE_TELEMETRY / DO_NOT_TRACK: MLflow 3 envia telemetria a un servidor externo;
# en un cluster sin salida a internet solo llena el log de errores de conexion.
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true

WORKDIR /app

# uv.lock + --frozen: se instalan exactamente las versiones bloqueadas, sin re-resolver.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
