# api/ — API de inferencia (FastAPI)

Servicio `api` del [`docker-compose.yaml`](../docker-compose.yaml). Sirve predicciones de especie de pingüino usando el modelo que está registrado en MLflow como **`penguins-classifier@champion`**. No tiene ningún modelo propio: todo lo obtiene de MLflow.

Swagger: **http://localhost:8000/docs** (o el puerto de `API_HOST_PORT`).

## Endpoints

| Método | Ruta | Descripción | Respuestas |
|---|---|---|---|
| `GET` | `/health` | Estado del servicio, modelo configurado y si hay uno cargado | 200 |
| `POST` | `/reload` | Fuerza la recarga del modelo desde MLflow | 200; 503 si MLflow no responde o no hay modelo |
| `POST` | `/predict` | Predice la especie de un pingüino | 200; 422 si la entrada es inválida; 503 si no hay modelo disponible |

### Ejemplo de predicción

```bash
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "island": "Torgersen", "bill_length_mm": 39.1, "bill_depth_mm": 18.7,
  "flipper_length_mm": 181, "body_mass_g": 3750, "sex": "male"}'
```

```json
{
  "species": "Adelie",
  "model_name": "penguins-classifier",
  "model_version": "1",
  "model_alias": "champion",
  "run_id": "43f00c1805fa48c8b69b42294c5dc2a1"
}
```

La respuesta incluye la **trazabilidad completa**: con el `run_id` se puede abrir en MLflow el run exacto que produjo el modelo, con sus hiperparámetros y métricas.

Otros ejemplos verificados:

| island | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g | sex | species |
|---|---:|---:|---:|---:|---|---|
| Torgersen | 39.1 | 18.7 | 181 | 3750 | male | Adelie |
| Biscoe | 47.5 | 15.0 | 218 | 4950 | female | Gentoo |
| Dream | 50.0 | 19.5 | 196 | 3900 | male | Chinstrap |

```bash
curl -s localhost:8000/health
# {"status":"ok","model":"penguins-classifier@champion","model_loaded":true,"model_version":"1"}
```

> 📸 **Imagen pendiente:** `docs/img/09-api-predict.png` — Swagger con una respuesta exitosa de `POST /predict`.

<!-- ![Predicción en Swagger](../docs/img/09-api-predict.png) -->

> 📸 **Imagen pendiente:** `docs/img/10-api-health.png` — respuesta de `GET /health` con `model_loaded: true`.

<!-- ![Health de la API](../docs/img/10-api-health.png) -->

## Archivos

| Archivo | Responsabilidad |
|---|---|
| `app/main.py` | Endpoints y `lifespan`: al arrancar intenta cargar el modelo, pero **nunca** impide que la API arranque (puede no haber ningún modelo registrado todavía) |
| `app/model_loader.py` | `MlflowModelLoader`: resuelve el alias, carga el modelo desde MLflow y lo mantiene en memoria |
| `app/schemas.py` | Contrato de entrada (`PenguinFeatures`) y salida (`PredictResponse`) con Pydantic; `FEATURE_COLUMNS` fija el orden de columnas |
| `pyproject.toml` | `fastapi`, `uvicorn`, `mlflow==3.16.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.4.6` |
| `Dockerfile` | `python:3.11-slim` + `uv` 0.12.10 + `uv sync --frozen` |

## Configuración del contenedor

| Variable | Valor | Para qué |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Servidor de MLflow (red interna) |
| `MODEL_NAME` | `penguins-classifier` | Modelo del Registry a servir |
| `MODEL_ALIAS` | `champion` | Alias que define qué versión servir |

- La API **no tiene credenciales de MinIO** ni `boto3`: descarga el modelo a través del proxy de artefactos de MLflow.
- En el compose arranca con `--reload` y con `./api/app` montado como volumen: los cambios en el código se aplican al guardar, sin reconstruir la imagen.
- Arranca después de que `mlflow` esté `healthy`.

## Cómo se carga el modelo

```
POST /predict
 └─ loader.predict(X)
      └─ refresh_if_needed()
           ├─ _latest_version()  → get_model_version_by_alias("penguins-classifier", "champion").version
           ├─ ¿distinta de la versión en memoria?
           │     sí → _load(): get_model_version_by_alias(...)
           │                   pyfunc.load_model("models:/penguins-classifier/<versión>")
           │     no → reutiliza el modelo en memoria
           └─ devuelve LoadedModel(model, name, version, alias, run_id)
```

1. **La API sigue un alias, no un número de versión.** Para promover un modelo nuevo basta con mover `champion` en MLflow: la siguiente petición lo detecta y lo carga, **sin redeploy ni reinicio**.
2. **Consultar es barato, descargar no.** En cada petición solo se pregunta a qué versión apunta el alias; el modelo se descarga únicamente cuando cambia.
3. **Se carga por número de versión** (`models:/<nombre>/<versión>`), no por alias. Si el alias se moviera entre la consulta y la descarga, cargar por alias podría traer la v2 mientras la respuesta reporta la v1. Así, la versión reportada siempre es la cargada. (En una URI, lo que va tras `@` es siempre un alias: `models:/penguins-classifier@1` busca un alias llamado `"1"` y falla).
4. **`threading.Lock`** protege la recarga: los endpoints con `def` corren en varios hilos, y sin el candado dos peticiones simultáneas podrían descargar el modelo dos veces. Es el mismo patrón que `ModelRegistry` de proyecto1, cambiando "el `.json` más reciente en MinIO" por "la versión del alias en MLflow".
5. **`mlflow.pyfunc`** en lugar de `mlflow.sklearn`: es la interfaz genérica de serving de MLflow y valida la entrada contra la `signature` del modelo.
6. **skops sin configuración extra:** la lista `skops_trusted_types` se guardó en el `MLmodel` al registrar, y MLflow la aplica al cargar.

## Validación de la entrada

La validación ocurre en dos capas:

| Capa | Qué valida | Error |
|---|---|---|
| Pydantic (`schemas.py`) | Campos obligatorios; `island` ∈ {Biscoe, Dream, Torgersen}; `sex` ∈ {female, male}; medidas > 0 | 422 con el detalle del campo |
| `pyfunc` (signature) | Columnas y tipos que espera el modelo | 503 con el mensaje de MLflow |

El modelo usa `OneHotEncoder(handle_unknown="ignore")`, así que una isla desconocida no lo haría fallar: la codificaría como todo ceros y **predeciría igual, sin avisar**. Por eso la lista cerrada de islas en Pydantic es importante: una entrada como `"island": "Marte"` se rechaza con 422 antes de llegar al modelo.

`/predict` convierte el JSON en un DataFrame de una fila con las columnas en el orden de `FEATURE_COLUMNS`, el mismo que `processed.penguins` y el entrenamiento.

## Notas

- **Aviso al cargar: `psutil (current: uninstalled, required: psutil==7.2.2)`.** MLflow infirió los requirements del modelo desde el entorno de Jupyter, donde `psutil` existe porque lo trae JupyterLab. El modelo no lo usa y las predicciones no se ven afectadas. Las librerías que sí importan (`scikit-learn`, `mlflow`, `pandas`, `numpy`) coinciden; si no, MLflow las listaría también.
- Si la API arranca sin modelo registrado, `/health` muestra `"model_loaded": false` y `/predict` responde 503. Al registrar el modelo, la siguiente petición lo carga sola.

```bash
cd taller004
docker compose logs -f api      # al arrancar: "Modelo inicial: penguins-classifier v1"
```
