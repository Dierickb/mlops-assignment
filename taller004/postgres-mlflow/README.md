# postgres-mlflow — Base de datos de metadata de MLflow

Servicio `postgres-mlflow` del [`docker-compose.yaml`](../docker-compose.yaml). Es el **backend store** del servidor de MLflow: guarda todo lo que MLflow sabe de los experimentos, **excepto los archivos** (modelos y demás artefactos), que van a [MinIO](../minio/README.md).

Esta carpeta solo contiene este README: el servicio usa la imagen oficial `postgres:16` sin archivos propios. Las tablas no se crean a mano; las crea el propio MLflow.

## Configuración

| Elemento | Valor |
|---|---|
| Imagen | `postgres:16` |
| Contenedor | `taller004_postgres_mlflow` |
| Base de datos | `${MLFLOW_DB_NAME}` (por defecto `mlflow`) |
| Usuario / contraseña | `${MLFLOW_DB_USER}` / `${MLFLOW_DB_PASSWORD}` |
| Puerto | `5432`, **solo en la red interna** (no se publica en el host) |
| Volumen | `mlflow-db-volume` → `/var/lib/postgresql/data` |
| Healthcheck | `pg_isready` cada 10 s |
| Quién se conecta | Solo el servidor `mlflow` |

El servidor de MLflow se conecta con:

```
postgresql+psycopg2://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@postgres-mlflow:5432/${MLFLOW_DB_NAME}
```

El `+psycopg2` es obligatorio: SQLAlchemy 2.1 usa psycopg v3 por defecto para `postgresql://`, y la imagen de MLflow solo tiene `psycopg2-binary`. Sin él, el servidor no arranca (ver [mlflow/README.md](../mlflow/README.md)).

## Por qué una base de datos dedicada

El enunciado pide separar la metadata de MLflow de los datos del negocio, y hay razones prácticas para hacerlo:

- **Ciclos de vida distintos.** Los datos de pingüinos se pueden recargar o reprocesar sin tocar el historial de experimentos, y al revés.
- **Esquema propio.** MLflow gestiona sus tablas con migraciones (Alembic). Mezclarlas con tablas de negocio arriesga conflictos de nombres y migraciones que tocan lo que no deben.
- **Respaldo y permisos por separado.** La metadata de experimentos (quién entrenó qué, con qué parámetros) suele tener requisitos de retención distintos a los datos.

La alternativa más simple, que MLflow guarde la metadata en un archivo SQLite dentro de su contenedor, se descartó porque se perdería al recrear el contenedor y no soporta bien accesos concurrentes.

## Qué guarda

Al arrancar por primera vez, MLflow ejecuta sus migraciones y crea unas 60 tablas (la tabla `alembic_version` registra en qué migración va). Las relevantes para este taller:

| Tabla | Contenido | Ejemplo en este taller |
|---|---|---|
| `experiments` | Experimentos | `penguinsTaller004` (id 1) y `Default` (id 0) |
| `runs` | Cada ejecución: estado, fechas, experimento | Los runs de la grilla (`rf-n100-d8`, …) |
| `params` | Parámetros por run (como texto) | `n_estimators = '100'`, `max_depth = '8'` |
| `metrics`, `latest_metrics` | Métricas por run | `cv_f1_macro_mean = 0.9909` |
| `tags` | Etiquetas por run | `mlflow.runName = rf-n100-d8` |
| `logged_models` | Modelos logueados (entidad nueva de MLflow 3) | `m-1202186e...` |
| `registered_models` | Modelos del Model Registry | `penguins-classifier` |
| `model_versions` | Versiones de cada modelo registrado y su run de origen | v1 → run `43f00c18...` |
| `registered_model_aliases` | Aliases | `champion` → v1 |

Las tablas de trazas, *guardrails*, *scorers*, etc. son de las funcionalidades de GenAI de MLflow y quedan vacías en este taller.

**Los parámetros se guardan como texto.** Por eso en `search_runs` los filtros de params van entre comillas (`params.max_depth = '8'`), y ordenar por un parámetro es alfabético (`"100" < "20"`). Las métricas son numéricas.

## Inspeccionar la base

```bash
cd taller004
docker compose exec postgres-mlflow psql -U mlflow -d mlflow
```

Consultas útiles:

```sql
-- Experimentos y cantidad de runs activos
SELECT e.name, count(r.run_uuid)
FROM experiments e LEFT JOIN runs r
  ON r.experiment_id = e.experiment_id AND r.lifecycle_stage = 'active'
GROUP BY e.name;

-- Modelo registrado, versión, alias y run de origen
SELECT mv.name, mv.version, a.alias, mv.run_id
FROM model_versions mv
LEFT JOIN registered_model_aliases a ON a.name = mv.name AND a.version = mv.version;
```

> 📸 **Imagen pendiente:** `docs/img/12-postgres-mlflow-tablas.png` — salida de las consultas anteriores.

<!-- ![Metadata de MLflow en Postgres](../docs/img/12-postgres-mlflow-tablas.png) -->

## Notas

- **Borrar desde la UI de MLflow es un borrado lógico**: el run o experimento pasa a `lifecycle_stage = 'deleted'` pero sigue en la tabla. Por eso un experimento borrado conserva su nombre reservado. Para eliminarlo físicamente existe `mlflow gc`.
- `docker compose down -v` borra el volumen y con él **todo el historial de experimentos y el Model Registry**. Los artefactos en MinIO quedarían huérfanos (también se borran con `-v`, porque MinIO tiene su propio volumen).
