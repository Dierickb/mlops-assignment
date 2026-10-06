# postgres-init/ — Base de datos de datos (`postgres-data`)

Servicio `postgres-data` del [`docker-compose.yaml`](../docker-compose.yaml). Guarda los **datos del negocio**: los pingüinos crudos y los procesados. Es una instancia distinta a la de MLflow ([postgres-mlflow](../postgres-mlflow/README.md)), como pide el enunciado.

Esta carpeta contiene el script que inicializa la base la primera vez que arranca.

## Configuración

| Elemento | Valor |
|---|---|
| Imagen | `postgres:16` |
| Contenedor | `taller004_postgres_data` |
| Base de datos | `${DATA_DB_NAME}` (por defecto `penguins`) |
| Usuario / contraseña | `${DATA_DB_USER}` / `${DATA_DB_PASSWORD}` |
| Puerto | `5432`, **solo en la red interna** |
| Volúmenes | `data-db-volume` (datos), `./postgres-init` → `/docker-entrypoint-initdb.d` (solo lectura), `./data` → `/data` (solo lectura) |
| Healthcheck | `pg_isready` cada 10 s |
| Quién se conecta | Solo `jupyter`, con `DATA_DB_URI` |

## Las dos etapas de los datos

| Esquema.tabla | Quién la crea | Quién la llena | Contenido | Filas |
|---|---|---|---|---|
| `raw.penguins` | `01-raw.sql` | `01-raw.sql` (`COPY` del CSV) | Copia fiel de `data/penguins.csv`, **con** valores faltantes | 344 |
| `processed.penguins` | El notebook (`DataFrame.to_sql`) | El notebook, sección 2 | Columnas del modelo + `species` + `rowid`, **sin** faltantes | 333 |

```
data/penguins.csv ──COPY──► raw.penguins ──notebook §2──► processed.penguins ──notebook §3──► entrenamiento
   (344 filas)              (344, con NULL)   (limpieza)    (333 filas)
```

### `raw`: los datos tal cual vienen

[`01-raw.sql`](01-raw.sql) crea los esquemas `raw` y `processed`, la tabla `raw.penguins` con las mismas columnas que el CSV, y la llena con:

```sql
COPY raw.penguins FROM '/data/penguins.csv' WITH (FORMAT csv, HEADER true, NULL 'NA');
```

- `NULL 'NA'` convierte los `NA` del CSV en `NULL`. Hay 11 filas con faltantes: 2 sin ninguna medida y 9 sin `sex`.
- No se limpia nada aquí: mantener el crudo intacto permite **reprocesar** si cambia la limpieza, sin volver a la fuente. Es el mismo criterio que la etapa `raw` de taller003 y proyecto1.
- El CSV es el mismo de taller001 (`data_sources/penguins.csv`). Se carga desde un archivo local en lugar de descargarlo de internet (como hacía taller003) para que el entorno no dependa de la red.

### `processed`: los datos listos para entrenar

La escribe el notebook (sección 2) con `if_exists="replace"`:

- Se quedan las 6 features (`island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`), el target `species` y `rowid`.
- Se eliminan las filas con faltantes en esas columnas (344 → 333). Se descarta en lugar de imputar porque son pocas (3 %) e imputar `sex` sería inventar un dato.
- `rowid` se conserva para rastrear cada fila procesada hasta su fila cruda.
- `island` y `sex` se guardan **como texto**: el encoding lo hace el pipeline del modelo (ver [README general, §6.1](../README.md#61-el-preprocesamiento-categórico-va-dentro-del-modelo)).
- `replace` hace que re-ejecutar el notebook deje la tabla consistente con la limpieza actual, en lugar de duplicar filas.

El entrenamiento (sección 3 del notebook) lee `processed.penguins` **desde la base de datos**, no desde el DataFrame en memoria: así se cumple que el modelo se entrena con los datos procesados almacenados.

## Cuándo se ejecuta `01-raw.sql`

**Una sola vez**: cuando el volumen `data-db-volume` se crea vacío. Es el mecanismo estándar de `/docker-entrypoint-initdb.d` de la imagen oficial de Postgres, que ejecuta en orden alfabético los `.sql` que encuentre (de ahí el prefijo `01-`).

Si se modifica el script o el CSV, el cambio **no se aplica** a un volumen existente. Hay que recrearlo:

```bash
docker compose down -v   # ⚠️ borra también experimentos, modelos y artefactos de MLflow
docker compose up -d
```

## Inspeccionar los datos

```bash
cd taller004
docker compose exec postgres-data psql -U data -d penguins
```

```sql
SELECT count(*) AS filas, count(*) FILTER (WHERE sex IS NULL) AS sin_sexo FROM raw.penguins;
SELECT species, count(*) FROM processed.penguins GROUP BY species;
```

Resultado esperado: 344 filas crudas (11 sin `sex`) y 146 Adelie, 119 Gentoo y 68 Chinstrap procesados.

> 📸 **Imagen pendiente:** `docs/img/11-postgres-data-tablas.png` — salida de las consultas anteriores.

<!-- ![Tablas raw y processed](../docs/img/11-postgres-data-tablas.png) -->
