# minio — Almacenamiento de artefactos de MLflow

Servicios `minio` y `minio-init` del [`docker-compose.yaml`](../docker-compose.yaml). MinIO es un almacenamiento de objetos compatible con S3, y aquí funciona como **artifact store** de MLflow: guarda los archivos de cada modelo (el modelo serializado, su `MLmodel`, sus requirements…). Está **dedicado a MLflow**: nada más escribe en él.

Esta carpeta solo contiene este README: ambos servicios usan imágenes sin archivos propios.

## Configuración

### `minio` (el servidor)

| Elemento | Valor |
|---|---|
| Imagen | `pgsty/silo:RELEASE.2026-09-16T00-00-00Z` |
| Contenedor | `taller004_minio` |
| Comando | `server /data --console-address ":9001"` |
| Credenciales | `${MINIO_ROOT_USER}` / `${MINIO_ROOT_PASSWORD}` |
| Puertos | `9000` API S3 → host `${MINIO_API_HOST_PORT}`; `9001` consola web → host `${MINIO_CONSOLE_HOST_PORT}` |
| Volumen | `minio-data-volume` → `/data` |
| Healthcheck | `mc ready local` cada 10 s |
| Quién se conecta | **Solo el servidor `mlflow`** |

### `minio-init` (crea el bucket)

| Elemento | Valor |
|---|---|
| Imagen | `pgsty/mc:RELEASE.2026-09-16T00-00-00Z` (cliente `mc`) |
| Arranca | Cuando `minio` está `healthy` |
| Hace | `mc mb --ignore-existing local/${MLFLOW_BUCKET}` y termina |
| Estado esperado | `Exited (0)` |

MLflow **no crea el bucket** si no existe: falla al subir el primer artefacto. Por eso `minio-init` lo crea antes, y el servidor de MLflow espera a que termine (`condition: service_completed_successfully`). `--ignore-existing` hace que sea seguro volver a ejecutarlo en cada `docker compose up`.

## Por qué Silo y no la imagen oficial

MinIO dejó de distribuir imágenes (retiró `minio/minio` y `minio/mc` de Docker Hub). **Silo** es un fork mantenido por PGSTY con el mismo código, el mismo protocolo S3 y las mismas variables `MINIO_*`. Es la misma decisión que en proyecto1. Se usa un **tag fijo**, no `:latest`, para que el entorno no cambie solo.

El healthcheck usa `mc ready local` porque las imágenes recientes no traen `curl`.

## Por qué solo MLflow tiene las credenciales

El servidor de MLflow arranca con `--serve-artifacts --artifacts-destination s3://mlflow`, es decir, hace de **proxy**:

```
jupyter / api ──HTTP (mlflow-artifacts:/...)──► mlflow ──S3 (boto3)──► minio
```

- Jupyter y la API nunca hablan con MinIO: suben y descargan a través de MLflow.
- Las credenciales (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) y el endpoint (`MLFLOW_S3_ENDPOINT_URL=http://minio:9000`) solo están en el contenedor de MLflow.
- Si mañana el almacenamiento cambia (por ejemplo, a S3 real), solo hay que reconfigurar el servidor.

## Cómo se organizan los archivos

```
mlflow/                                   ← bucket
└── 1/                                    ← experiment_id (penguinsTaller004)
    └── models/
        └── m-1202186e752d4fa184e72a13d598c784/   ← logged model (MLflow 3)
            └── artifacts/
                ├── MLmodel                    ← descriptor: flavors, signature, skops_trusted_types
                ├── model.skops                ← el pipeline serializado (~2.4 MB)
                ├── requirements.txt           ← dependencias pip del modelo
                ├── python_env.yaml
                ├── conda.yaml
                ├── input_example.json         ← 3 filas de ejemplo
                └── serving_input_example.json ← ejemplo con el formato de petición de serving
```

En MLflow 3, los modelos son una entidad propia (*logged model*, con id `m-...`) y sus archivos viven bajo `models/`, no bajo el run. El modelo `m-1202186e...` es el registrado como `penguins-classifier` v1.

## Consola web

http://localhost:9001 (o el puerto de `MINIO_CONSOLE_HOST_PORT`), con `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`.

> 📸 **Imagen pendiente:** `docs/img/08-minio-bucket.png` — bucket `mlflow`, carpeta `1/models/m-.../artifacts/` con los archivos del modelo.

<!-- ![Artefactos del modelo en MinIO](../docs/img/08-minio-bucket.png) -->

Desde la terminal:

```bash
cd taller004
set -a && . ./.env && set +a
docker run --rm --network taller004_default --entrypoint /bin/sh \
  pgsty/mc:RELEASE.2026-09-16T00-00-00Z -c \
  "mc alias set l http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD >/dev/null && mc du l/$MLFLOW_BUCKET"
```

## Notas

- **Choque de puertos con proyecto1.** Si otro MinIO ya ocupa el 9000, Docker no puede publicar el puerto y deja el contenedor **sin red**: `minio-init` falla con `lookup minio ... no such host`. Solución: cambiar `MINIO_API_HOST_PORT` y `MINIO_CONSOLE_HOST_PORT` en `.env`.
- **Borrar un run en MLflow no borra sus archivos en MinIO.** El borrado de MLflow es lógico; los artefactos quedan en el bucket hasta ejecutar `mlflow gc`.
