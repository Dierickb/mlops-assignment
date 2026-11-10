-- Esquemas y tablas de la base de DATOS del proyecto (no la de Airflow ni la de MLflow).
-- Se ejecuta automaticamente UNA vez, cuando el PVC de postgres-data se inicializa
-- (mecanismo /docker-entrypoint-initdb.d de la imagen oficial). Todo es idempotente
-- (IF NOT EXISTS), asi que el DAG tambien puede ejecutarlo antes de cargar datos.

-- Tres capas separadas, como pide el enunciado (seccion 4.3):
CREATE SCHEMA IF NOT EXISTS raw;        -- datos tal como llegan del archivo fuente
CREATE SCHEMA IF NOT EXISTS clean;      -- datos limpios y transformados para entrenar
CREATE SCHEMA IF NOT EXISTS inference;  -- un registro por cada prediccion de la API

-- ---------------------------------------------------------------------------
-- RAW
-- ---------------------------------------------------------------------------

-- Control de lotes: una fila por lote de carga (max 15.000 registros).
-- Permite saber que filas del archivo ya se cargaron y reanudar sin duplicar.
CREATE TABLE IF NOT EXISTS raw.load_batches (
    batch_id      INTEGER      PRIMARY KEY,
    source_file   TEXT         NOT NULL,
    start_row     INTEGER      NOT NULL,          -- fila inicial del archivo (inclusiva)
    end_row       INTEGER      NOT NULL,          -- fila final (exclusiva)
    n_rows        INTEGER      NOT NULL,
    status        TEXT         NOT NULL CHECK (status IN ('loading', 'loaded', 'failed')),
    started_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);

-- Registros crudos. `data` guarda la fila completa del CSV como JSONB, sin
-- transformaciones destructivas: si cambia la limpieza, se reprocesa desde aqui.
CREATE TABLE IF NOT EXISTS raw.diabetes (
    id              BIGSERIAL    PRIMARY KEY,
    batch_id        INTEGER      NOT NULL REFERENCES raw.load_batches (batch_id),
    source_row      INTEGER      NOT NULL,        -- posicion de la fila en el archivo fuente
    source_file     TEXT         NOT NULL,
    load_timestamp  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    row_hash        CHAR(64)     NOT NULL,        -- SHA-256 del contenido de la fila
    status          TEXT         NOT NULL DEFAULT 'loaded'
                                 CHECK (status IN ('loaded', 'valid', 'rejected')),
    data            JSONB        NOT NULL,
    -- Control de duplicados: la misma fila del mismo archivo no entra dos veces,
    -- aunque el DAG se reejecute (INSERT ... ON CONFLICT DO NOTHING).
    CONSTRAINT uq_raw_diabetes_row UNIQUE (source_file, source_row, row_hash)
);
CREATE INDEX IF NOT EXISTS ix_raw_diabetes_batch ON raw.diabetes (batch_id);

-- ---------------------------------------------------------------------------
-- INFERENCE: registro de cada prediccion (seccion 4.3 del enunciado)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference.predictions (
    request_id      UUID              PRIMARY KEY,
    requested_at    TIMESTAMPTZ       NOT NULL DEFAULT now(),
    input           JSONB             NOT NULL,   -- datos de entrada recibidos por la API
    prediction      TEXT,                         -- clase predicha
    score           DOUBLE PRECISION,             -- probabilidad de la clase predicha
    probabilities   JSONB,                        -- probabilidad por clase
    model_name      TEXT,
    model_version   TEXT,
    model_alias     TEXT,
    latency_ms      DOUBLE PRECISION,             -- tiempo de respuesta de la inferencia
    status          TEXT              NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'error')),
    error           TEXT
);
CREATE INDEX IF NOT EXISTS ix_inference_requested_at ON inference.predictions (requested_at);

-- Las tablas de CLEAN las crea el DAG (pandas.to_sql) porque sus columnas dependen
-- de la ingenieria de caracteristicas.
