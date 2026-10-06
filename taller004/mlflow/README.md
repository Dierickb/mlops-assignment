# mlflow/ — Servidor de tracking de MLflow

Servicio `mlflow` del [`docker-compose.yaml`](../docker-compose.yaml). Es el centro del taller: recibe y guarda todo lo que produce la experimentación (runs, parámetros, métricas y modelos), mantiene el **Model Registry** y entrega los modelos a la API.

UI: **http://localhost:5000** (o el puerto de `MLFLOW_HOST_PORT`; en macOS, `5001`).

## Archivos

| Archivo | Responsabilidad |
|---|---|
| `Dockerfile` | `python:3.11-slim` + binario de `uv` 0.12.10 + `uv sync --frozen`. No define `CMD`: el comando de arranque está en el compose, junto al resto de la configuración |
| `pyproject.toml` | `mlflow==3.16.1`, `psycopg2-binary==2.9.13` (driver de Postgres), `boto3==1.43.108` (cliente S3 para MinIO) |
| `uv.lock` | Todas las dependencias transitivas bloqueadas |

## Cómo arranca

```bash
uv run --no-sync mlflow server \
  --backend-store-uri postgresql+psycopg2://<user>:<pass>@postgres-mlflow:5432/<db> \
  --artifacts-destination s3://${MLFLOW_BUCKET} \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}"
```

| Opción | Qué hace | Por qué |
|---|---|---|
| `--backend-store-uri` | Dónde guarda la metadata | En [postgres-mlflow](../postgres-mlflow/README.md), una base dedicada |
| `+psycopg2` en la URI | Fuerza el driver psycopg2 | SQLAlchemy 2.1 usa psycopg v3 por defecto, que no está instalado. Sin esto: `ModuleNotFoundError: No module named 'psycopg'` |
| `--artifacts-destination s3://mlflow` | Dónde guarda los archivos | En el bucket de [MinIO](../minio/README.md) |
| `--serve-artifacts` | El servidor hace de **proxy** de artefactos | Los clientes no necesitan credenciales de MinIO (ver abajo) |
| `--host 0.0.0.0` | Escucha en todas las interfaces | Para ser alcanzable desde otros contenedores y desde el host |
| `--allowed-hosts` | Lista de headers `Host` aceptados | Seguridad de MLflow 3 (ver abajo) |
| `uv run --no-sync` | Ejecuta en el entorno ya instalado | Evita que `uv` intente resincronizar en cada arranque |

Variables de entorno del contenedor, que usa `boto3` dentro del servidor para hablar con MinIO:

| Variable | Valor |
|---|---|
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `${MINIO_ROOT_USER}` / `${MINIO_ROOT_PASSWORD}` |

**Dependencias de arranque:** espera a que `postgres-mlflow` esté `healthy` y a que `minio-init` haya creado el bucket. **Healthcheck:** `GET /health` con Python (`urllib`), porque la imagen slim no trae `curl`. Jupyter y la API esperan a este healthcheck antes de arrancar.

## Las tres piezas que coordina

```
                    ┌──────────────► postgres-mlflow   (qué pasó: experimentos, runs, params,
                    │                                   métricas, Model Registry, aliases)
cliente ──HTTP──► mlflow
(jupyter, api)      │
                    └──────────────► minio/mlflow      (los archivos: model.skops, MLmodel,
                                                        requirements.txt, ...)
```

### El proxy de artefactos

Con `--serve-artifacts`, la ubicación de artefactos de cada experimento es `mlflow-artifacts:/<experiment_id>`. Cuando el notebook llama a `log_model`, sube los archivos **al servidor de MLflow por HTTP**, y el servidor los escribe en MinIO.

- Las credenciales de MinIO solo existen en este contenedor.
- Jupyter y la API solo necesitan `MLFLOW_TRACKING_URI=http://mlflow:5000`. Ni siquiera tienen `boto3` instalado.
- **Alternativa descartada:** `--default-artifact-root s3://mlflow`, donde cada cliente escribe directo a S3. Obliga a repartir las credenciales y a instalar `boto3` en todos los contenedores.

### Seguridad: `--allowed-hosts`

MLflow 3.x valida el header `Host` de cada petición para prevenir **DNS rebinding** (un ataque en el que una web maliciosa engaña al navegador para que hable con un servicio interno). Por defecto solo acepta `localhost` e IPs privadas, así que las peticiones de Jupyter y la API, que usan el nombre `mlflow:5000`, eran rechazadas.

`MLFLOW_ALLOWED_HOSTS` (en `.env`) declara explícitamente los válidos:

```
mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*,10.*,192.168.*
```

- `mlflow`, `mlflow:5000`: el nombre del servicio en la red de Docker.
- `localhost:*`, `127.0.0.1:*`: el navegador en la misma máquina, con cualquier puerto (por ejemplo 5001 en macOS).
- `10.*`, `192.168.*`: acceso por IP desde la red local o una MV.

Al pasar la lista explícita se reemplaza el valor por defecto, por eso se repiten `localhost` y las IPs privadas.

El **MLflow Assistant** de la UI tampoco funciona en este montaje: solo responde si el navegador está en la misma máquina que el servidor, y aquí el servidor corre dentro de un contenedor.

## Qué hay en la UI

| Sección | Qué muestra en este taller |
|---|---|
| **Experiments → `penguinsTaller004` → Runs** | Los runs de la grilla, con columnas de params y métricas |
| **Run → Overview** | Params (`n_estimators`, `max_depth`) y métricas (`accuracy`, `f1_macro`, `cv_f1_macro_mean`, `cv_f1_macro_std`) |
| **Run → Artifacts** | El modelo logueado: `MLmodel`, `model.skops`, `requirements.txt`… |
| **Compare** (varios runs seleccionados) | Gráfico de coordenadas paralelas hiperparámetros → métrica |
| **Models** (dentro del experimento) | Los *logged models* de MLflow 3 (`m-...`), cada uno vinculado a su run |
| **Model registry** | `penguins-classifier`, versión 1, alias `@champion` |

> 📸 **Imagen pendiente:** `docs/img/03-mlflow-runs.png` — tabla de runs del experimento.

<!-- ![Runs del experimento](../docs/img/03-mlflow-runs.png) -->

> 📸 **Imagen pendiente:** `docs/img/04-mlflow-compare.png` — vista *Compare* de los 25 runs.

<!-- ![Comparación de runs](../docs/img/04-mlflow-compare.png) -->

> 📸 **Imagen pendiente:** `docs/img/06-mlflow-run-artifacts.png` — pestaña *Artifacts* del run elegido.

<!-- ![Artefactos del run](../docs/img/06-mlflow-run-artifacts.png) -->

> 📸 **Imagen pendiente:** `docs/img/07-mlflow-model-registry.png` — `penguins-classifier` v1 `@champion`.

<!-- ![Model Registry](../docs/img/07-mlflow-model-registry.png) -->

## Conceptos de MLflow usados en el taller

| Concepto | Qué es | En este taller |
|---|---|---|
| **Experimento** | Agrupa runs relacionados | `penguinsTaller004` |
| **Run** | Una ejecución de entrenamiento: params, métricas, tags | `rf-n100-d8`, id `43f00c18...` |
| **Logged model** | Un modelo guardado por un run (MLflow 3) | `m-1202186e...` |
| **Registered model** | Nombre en el catálogo, con versiones | `penguins-classifier` |
| **Versión** | Un logged model concreto dentro del registered model | v1 |
| **Alias** | Etiqueta móvil que apunta a una versión | `champion` → v1 |

URIs que se usan para referirse a un modelo:

| URI | Significa |
|---|---|
| `runs:/<run_id>/model` | El modelo con `name="model"` guardado por ese run (forma clásica; MLflow 3 la traduce al logged model) |
| `models:/<model_id>` | Un logged model por su id (forma nativa de MLflow 3) |
| `models:/penguins-classifier/1` | La versión 1 del modelo registrado |
| `models:/penguins-classifier@champion` | La versión a la que apunte hoy el alias `champion` |

## Comandos útiles

```bash
cd taller004
docker compose logs -f mlflow                    # logs del servidor
curl -s localhost:5000/health                    # "OK" (ajustar el puerto si es 5001)
docker compose up -d --build mlflow              # reconstruir tras cambiar pyproject.toml
```
