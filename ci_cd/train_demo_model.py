"""Entrena un modelo DE DEMOSTRACION para poder probar la imagen runtime.

OJO: usa datos sinteticos. No sirve para predecir nada real; solo permite
verificar que la API carga un Pipeline de sklearn y responde de punta a punta.
En el Proyecto 2 el modelo real lo entrena Airflow y lo sirve MLflow.

Uso:  python train_demo_model.py models/model.joblib
"""

import random
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.preprocessing import CATEGORICAL_FEATURES, FEATURE_ORDER, NUMERIC_RANGES, build_features


def synthetic_records(n: int, seed: int = 42) -> tuple[pd.DataFrame, list[int]]:
    """Genera n registros validos y una etiqueta que depende de number_inpatient."""
    rng = random.Random(seed)  # semilla fija -> mismo dataset siempre (repetible)
    rows, labels = [], []
    for _ in range(n):
        low = rng.randrange(0, 100, 10)
        raw = {"age": f"[{low}-{low + 10})"}
        for name, (lo, hi) in NUMERIC_RANGES.items():
            raw[name] = rng.randint(lo, hi)
        raw.update(
            race=rng.choice(["Caucasian", "AfricanAmerican", "?"]),
            gender=rng.choice(["Female", "Male"]),
            A1Cresult=rng.choice(["None", ">8", "Norm"]),
            insulin=rng.choice(["No", "Steady", "Up"]),
            change=rng.choice(["Ch", "No"]),
            diabetesMed=rng.choice(["Yes", "No"]),
        )
        rows.append(build_features(raw))  # la MISMA funcion que usa la API
        labels.append(int(raw["number_inpatient"] >= 3))
    return pd.DataFrame(rows, columns=FEATURE_ORDER), labels


def train(n: int = 2000, seed: int = 42) -> Pipeline:
    X, y = synthetic_records(n, seed)
    numeric = [c for c in FEATURE_ORDER if c not in CATEGORICAL_FEATURES]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/model.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(train(), out)
    print(f"modelo demo guardado en {out}")
