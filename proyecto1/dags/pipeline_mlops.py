from __future__ import annotations

import os
import io
import json
import pickle
from datetime import datetime, timedelta

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from minio import Minio
from airflow import DAG
from airflow.operators.python import PythonOperator


DB_URI = os.environ["MLOPS_DB_URI"]
EXTERNAL_API_BASE_URL = os.environ["EXTERNAL_API_BASE_URL"]
GRUPO_NUMERO = os.environ["GRUPO_NUMERO"]
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ROOT_USER = os.environ["MINIO_ROOT_USER"]
MINIO_ROOT_PASSWORD = os.environ["MINIO_ROOT_PASSWORD"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
COVERTYPE_NUMERIC_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
COVERTYPE_ALL_COLUMNS = COVERTYPE_NUMERIC_COLUMNS + ["Wilderness_Area", "Soil_Type", "Cover_Type"]
FEATURE_COLUMNS = COVERTYPE_NUMERIC_COLUMNS + ["Wilderness_Area", "Soil_Type"]
TARGET_COLUMN = "Cover_Type"
MIN_TRAIN_ROWS = 50
default_args = {
    "owner": "grupo7",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def _get_engine():
    return create_engine(DB_URI, echo=False)

def _get_minio():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )

def fetch_batch_grupo7(**context):
    response = requests.get(
        f"{EXTERNAL_API_BASE_URL}/data",
        params={"group_number": GRUPO_NUMERO},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO raw.batches (fetched_at, group_number, payload) "
                "VALUES (:fetched_at, :group_number, :payload)"
            ),
            {
                "fetched_at": datetime.utcnow(),
                "group_number": payload["group_number"],
                "payload": json.dumps(payload),
            },
        )
    print(f"Batch #{payload['batch_number']} guardado ({len(payload['data'])} filas) para grupo {payload['group_number']}.")

def preprocess(**context):
    engine = _get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, payload FROM raw.batches ORDER BY fetched_at DESC LIMIT 1")
        ).fetchone()

    if row is None:
        raise ValueError("No hay datos en raw.batches todavia.")

    raw_id, payload = row
    if isinstance(payload, str):
        payload = json.loads(payload)

    df = pd.DataFrame(payload["data"], columns=COVERTYPE_ALL_COLUMNS)

    for col in COVERTYPE_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Cover_Type"] = pd.to_numeric(df["Cover_Type"], errors="coerce")

    df = df.dropna(subset=COVERTYPE_NUMERIC_COLUMNS + ["Cover_Type"])
    df["source_raw_id"] = raw_id

    with engine.begin() as conn:
        df.to_sql("covertype_clean", con=conn, schema="processed", if_exists="append", index=False)

    print(f"{len(df)} registros preprocesados (raw_id={raw_id}) escritos en processed.covertype_clean.")

def build_training_set(**context):
    engine = _get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM processed.covertype_clean"), conn)

    if df.empty:
        raise ValueError("processed.covertype_clean esta vacia; nada que preparar.")

    total = len(df)
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].drop_duplicates()

    with engine.begin() as conn:
        df.to_sql("covertype_train", con=conn, schema="train_ready",
                  if_exists="replace", index=False)

    print(f"train_ready.covertype_train: {len(df)} filas unicas (de {total} acumuladas).")

def train_model(**context):
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    engine = _get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM train_ready.covertype_train"), conn)

    n_clases = df[TARGET_COLUMN].nunique()
    if len(df) < MIN_TRAIN_ROWS or n_clases < 2:
        raise ValueError(
            f"Datos insuficientes para entrenar: {len(df)} filas, {n_clases} clases. "
            "Se conserva el modelo anterior en MinIO."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("prep", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), ["Wilderness_Area", "Soil_Type"])],
            remainder="passthrough",
        )),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    model_name = f"covertype_rf_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    # Serializamos con pickle (implementacion en C), NO con joblib.dump.
    # Airflow importa `dill` al arrancar, y dill registra sus propias funciones en el pickle
    # de Python puro. joblib copia esa tabla (NumpyPickler.dispatch) al importarse, o sea
    # DESPUES de dill, asi que joblib.dump deja referencias a dill en el modelo y la API
    # (que no tiene dill) no puede cargarlo. El Pickler en C no usa esa tabla.
    # Protocolo 4: el mas alto que soporta Python 3.7 (Airflow); Python 3.10 (API) lo lee.
    # joblib.load (en la API) lee sin problema un pickle estandar.
    model_buffer = io.BytesIO()
    pickle.dump(model, model_buffer, protocol=4)
    model_buffer.seek(0)

    # Validar el artefacto ANTES de publicarlo: nunca subir un modelo que la API no pueda cargar.
    if b"dill" in model_buffer.getvalue():
        raise RuntimeError("El modelo serializado depende de dill; no se sube a MinIO.")

    metadata = {
        "model_name": model_name,
        "model_file": f"{model_name}.pkl",
        "created_at": datetime.utcnow().isoformat(),
        "accuracy": round(float(accuracy), 4),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "features": FEATURE_COLUMNS,
        "classes": sorted(int(c) for c in y.unique()),
        "sklearn_version": sklearn.__version__,
    }
    meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")

    client = _get_minio()
    client.put_object(
        MINIO_BUCKET, f"{model_name}.pkl", model_buffer,
        length=model_buffer.getbuffer().nbytes, content_type="application/octet-stream",
    )
    client.put_object(
        MINIO_BUCKET, f"{model_name}.json", io.BytesIO(meta_bytes),
        length=len(meta_bytes), content_type="application/json",
    )
    print(f"Modelo {model_name} subido a MinIO (accuracy={accuracy:.4f}).")

with DAG(
    dag_id="pipeline_mlops_grupo7",
    description="Ingesta por lotes -> preprocesamiento -> set de entrenamiento -> modelo en MinIO",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["proyecto1", "grupo7"],
) as dag:

    t_fetch = PythonOperator(task_id="fetch_batch", python_callable=fetch_batch_grupo7,retries=0)
    t_preprocess = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_build = PythonOperator(task_id="build_training_set", python_callable=build_training_set)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)

    t_fetch >> t_preprocess >> t_build >> t_train