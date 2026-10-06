# jupyter/ — JupyterLab y el notebook de experimentación

Servicio `jupyter` del [`docker-compose.yaml`](../docker-compose.yaml). JupyterLab instalado con `uv`, donde vive el notebook que procesa los datos, ejecuta la experimentación y registra el modelo en MLflow.

URL: **http://localhost:8888** (o el puerto de `JUPYTER_HOST_PORT`), token `${JUPYTER_TOKEN}` (por defecto `devtoken`).

## Archivos

| Archivo | Responsabilidad |
|---|---|
| `Dockerfile` | `python:3.11-slim` + `uv` 0.12.10 + `uv sync --frozen`; arranca con `uv run --no-sync jupyter lab` |
| `pyproject.toml` | `jupyterlab==4.6.4`, `mlflow==3.16.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.4.6`, `sqlalchemy==2.1.3`, `psycopg2-binary==2.9.13` |
| `uv.lock` | Dependencias transitivas bloqueadas |
| `notebooks/experimentos_mlflow.ipynb` | El notebook del taller |

Las versiones de `mlflow`, `scikit-learn`, `pandas` y `numpy` son **idénticas** a las de [`api/`](../api/README.md): el modelo se serializa aquí y se deserializa allá.

## Configuración del contenedor

| Elemento | Valor |
|---|---|
| Volumen | `./jupyter/notebooks` → `/app/notebooks`: lo que se edita en JupyterLab queda en el repositorio |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` |
| `DATA_DB_URI` | `postgresql+psycopg2://...@postgres-data:5432/...` |
| `MODEL_NAME`, `MODEL_ALIAS` | `penguins-classifier`, `champion` (los mismos que lee la API) |
| `GIT_PYTHON_REFRESH` | `quiet`: la imagen no trae git y MLflow avisaría en cada run |
| Arranca después de | `postgres-data` `healthy` y `mlflow` `healthy` |

**El notebook usa hostnames de la red de Docker** (`mlflow`, `postgres-data`). Por eso se ejecuta en JupyterLab, dentro del contenedor. Si se abre con un kernel local en VS Code, las celdas que se conectan fallan.

`MLFLOW_TRACKING_URI` como variable de entorno ya configura el cliente: MLflow la lee automáticamente al importarse, así que el notebook habla con el servidor aunque no llame explícitamente a `mlflow.set_tracking_uri(...)`.

## El notebook, sección por sección

| § | Sección | Qué hace | Resultado |
|---|---|---|---|
| 0 | Configuración | Lee `DATA_DB_URI`, `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS`; define `EXPERIMENT_NAME = "penguinsTaller004"`; crea el engine de SQLAlchemy | — |
| 1 | Datos crudos | `SELECT * FROM raw.penguins` | 344 filas, 11 con faltantes |
| 2 | Preprocesamiento | Selecciona features + target + `rowid`, elimina faltantes, castea numéricas, escribe `processed.penguins` (`replace`) | 333 filas |
| 3 | Train/test | Lee `processed.penguins` **desde la BD**; `train_test_split` 80/20 estratificado, `random_state=42` | 266 / 67 |
| 4 | MLflow | `mlflow.set_experiment(EXPERIMENT_NAME)` | Experimento id 1 |
| 5 | `train_and_log` | Define la función que entrena y registra un run | — |
| 6 | Experimentación | Grilla 5 × 5 y loop de `train_and_log` | 25 runs |
| 7 | Registro | `search_runs` → elegir → `register_model` → alias | `penguins-classifier` v1 `@champion` |
| 8 | Verificación | `pyfunc.load_model("models:/penguins-classifier@champion")` y predicción | Especies como texto |

Para reproducirlo: **Run → Run All Cells**. Si se reinicia el kernel, hay que volver a ejecutar desde la sección 0 (las funciones usan variables globales como `X_train`).

### El modelo (sección 5)

```
Pipeline
├── "preprocess": ColumnTransformer
│     ├── OneHotEncoder(handle_unknown="ignore") → island, sex
│     └── passthrough                            → bill_length_mm, bill_depth_mm,
│                                                  flipper_length_mm, body_mass_g
└── "model": RandomForestClassifier(**params, random_state=42, n_jobs=-1)
```

El encoding va **dentro** del pipeline, así que el modelo registrado recibe los 6 campos crudos, igual que la API.

### Qué registra cada run (`train_and_log`)

| Tipo | Qué | Cómo |
|---|---|---|
| Params | `n_estimators`, `max_depth` | `mlflow.log_params(params)`: el mismo diccionario que alimenta al modelo, así no puede registrarse un valor distinto del usado |
| Métricas de test | `accuracy`, `f1_macro` | Sobre `X_test` (67 filas) |
| Métricas de CV | `cv_f1_macro_mean`, `cv_f1_macro_std` | `cross_val_score` con `StratifiedKFold(5, shuffle=True, random_state=42)` sobre el **pipeline completo** y solo con train |
| Modelo | El pipeline entero | `mlflow.sklearn.log_model(name="model", signature=..., input_example=X_train.head(3), skops_trusted_types=["sklearn.tree._tree.Tree"])` |

- **`signature`**: se infiere con `infer_signature(X_train, pipeline.predict(X_train))`. Es el contrato de columnas y tipos; `pyfunc` lo usa para validar la entrada.
- **`skops_trusted_types`**: MLflow 3.16 serializa con skops (más seguro que pickle) y exige declarar los tipos no estándar. Se declara solo `sklearn.tree._tree.Tree`, la estructura interna de los árboles. La lista queda guardada en el `MLmodel`, así que la API la aplica al cargar.
- **`run_name`**: `rf-n{n_estimators}-d{max_depth}`, para distinguir los runs en la UI.
- La función devuelve `run.info.run_id` (el identificador único que genera MLflow), no el nombre: los nombres pueden repetirse.

### La grilla (sección 6)

`n_estimators ∈ {1, 5, 20, 100, 300}` × `max_depth ∈ {1, 2, 4, 8, None}` = 25 combinaciones, con `itertools.product`. El rango incluye valores deliberadamente pobres para que la experimentación muestre el efecto de cada hiperparámetro (resultados en el [README general, §7](../README.md#7-resultados-de-la-experimentación)).

### Selección y registro (sección 7)

1. **Ranking:** `search_runs` con `order_by=["metrics.cv_f1_macro_mean DESC", "metrics.cv_f1_macro_std ASC"]`. Se consulta a MLflow, la fuente de verdad, y no al DataFrame en memoria, que se pierde al reiniciar el kernel.
2. **Desempate:** 4 runs empatan en 0.9909 ± 0.0112. Se elige el más simple con `filter_string="params.n_estimators = '100' and params.max_depth = '8'"`, más reciente primero (`attributes.start_time DESC`). Los params van entre comillas porque MLflow los guarda como texto.
3. **Registro:** `mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)` → versión 1.
4. **Alias:** `MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)`.

Los prefijos de `search_runs`: `metrics.` (lo registrado con `log_metrics`), `params.` (con `log_params`), `tags.` (por ejemplo `tags.mlflow.runName`) y `attributes.` (datos del run: `start_time`, `status`…).

> 📸 **Imagen pendiente:** `docs/img/02-jupyter-notebook.png` — el notebook en JupyterLab.

<!-- ![Notebook en JupyterLab](../docs/img/02-jupyter-notebook.png) -->

> 📸 **Imagen pendiente:** `docs/img/05-mlflow-run-detalle.png` — el run elegido en MLflow, con params y métricas.

<!-- ![Run elegido](../docs/img/05-mlflow-run-detalle.png) -->

## Notas

- **Volver a ejecutar la celda de la grilla crea 25 runs nuevos**, con los mismos nombres e ids distintos. Los resultados son idénticos (`random_state=42`), pero el experimento se llena de duplicados.
- **Aviso `Failed to resolve installed pip version`**: el entorno lo creó `uv`, que no instala `pip`. MLflow deja `pip` sin versión en `conda.yaml`; `requirements.txt` no se ve afectado.
- **Agregar una librería:** `cd jupyter && uv add <paquete>` y luego `docker compose up -d --build jupyter`. No usar `pip install` dentro del contenedor: se pierde al recrearlo.
