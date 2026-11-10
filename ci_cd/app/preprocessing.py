"""Preprocesamiento de un registro del dataset Diabetes 130-US hospitals.

Este modulo NO conoce FastAPI ni el modelo: solo transforma datos.
Por eso se puede probar directamente (sin HTTP) y es reutilizable en
el DAG de Airflow y en la API del Proyecto 2: la misma funcion en
entrenamiento y en serving evita el "training/serving skew".
"""

import re

# Valores que el dataset usa para "no se sabe". En el CSV original aparecen
# como "?" (race, payer_code, medical_specialty) y "Unknown/Invalid" (gender).
MISSING_TOKENS = {"", "?", "none", "null", "nan", "unknown", "unknown/invalid"}
UNKNOWN = "Unknown"

# Rango permitido (min, max) de cada variable numerica.
# Los limites salen del dataset original (min/max observados), redondeados.
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (0, 150),
    "num_procedures": (0, 6),
    "num_medications": (0, 100),
    "number_outpatient": (0, 50),
    "number_emergency": (0, 100),
    "number_inpatient": (0, 30),
    "number_diagnoses": (1, 16),
}

CATEGORICAL_FEATURES = ["race", "gender", "A1Cresult", "insulin", "change", "diabetesMed"]

# Orden EXACTO de las columnas que recibe el modelo. Si este orden cambia,
# el modelo recibe las columnas desordenadas y predice basura sin fallar:
# por eso hay una prueba que lo fija.
FEATURE_ORDER = ["age_years", *NUMERIC_RANGES.keys(), *CATEGORICAL_FEATURES]

_AGE_BRACKET = re.compile(r"^\[(\d{1,3})-(\d{1,3})\)$")


def clean_category(value: object) -> str:
    """Limpieza de texto: quita espacios, unifica los faltantes en 'Unknown'.

    >>> clean_category("  Caucasian ")
    'Caucasian'
    >>> clean_category("?")
    'Unknown'
    """
    if value is None:
        return UNKNOWN
    text = " ".join(str(value).split())  # recorta y colapsa espacios internos
    if text.lower() in MISSING_TOKENS:
        return UNKNOWN
    return text


def age_bracket_to_years(bracket: str) -> int:
    """Transformacion de feature: '[70-80)' -> 75 (punto medio del rango).

    El dataset trae la edad como texto en rangos de 10 anios. El modelo
    necesita un numero; el punto medio conserva el orden entre rangos.
    """
    match = _AGE_BRACKET.match(str(bracket).strip())
    if not match:
        raise ValueError(f"rango de edad invalido: {bracket!r} (se espera '[70-80)')")
    low, high = int(match.group(1)), int(match.group(2))
    if not 0 <= low < high <= 100:
        raise ValueError(f"rango de edad fuera de [0, 100]: {bracket!r}")
    return (low + high) // 2


def validate_ranges(record: dict) -> dict[str, int]:
    """Validacion de rangos numericos.

    Devuelve solo las variables numericas, convertidas a int.
    - KeyError si falta alguna (el mensaje dice cual).
    - ValueError si alguna esta fuera de rango o no es entera.
    """
    missing = [name for name in NUMERIC_RANGES if name not in record]
    if missing:
        raise KeyError(f"faltan columnas: {missing}")

    clean: dict[str, int] = {}
    for name, (low, high) in NUMERIC_RANGES.items():
        raw = record[name]
        if isinstance(raw, bool):  # bool es subclase de int en Python: rechazarlo explicito
            raise ValueError(f"{name} debe ser numerico, llego {raw!r}")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} debe ser numerico, llego {raw!r}") from None
        if not number.is_integer():
            raise ValueError(f"{name} debe ser entero, llego {raw!r}")
        if not low <= number <= high:
            raise ValueError(f"{name}={raw} fuera de rango [{low}, {high}]")
        clean[name] = int(number)
    return clean


def build_features(record: dict) -> dict:
    """Combina las tres funciones anteriores en el vector que recibe el modelo.

    Devuelve un dict con las claves en el orden de FEATURE_ORDER.
    """
    if "age" not in record:
        raise KeyError("faltan columnas: ['age']")
    features: dict = {"age_years": age_bracket_to_years(record["age"])}
    features.update(validate_ranges(record))
    for name in CATEGORICAL_FEATURES:
        features[name] = clean_category(record.get(name))
    return {name: features[name] for name in FEATURE_ORDER}


def probability_to_risk(probability: float) -> str:
    """Convierte la probabilidad del modelo en una etiqueta de negocio.

    Mismo patron que classify_score del demo del curso. Umbrales:
    >= 0.6 'high', >= 0.3 'medium', resto 'low'.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"la probabilidad debe estar entre 0 y 1, llego {probability}")
    if probability >= 0.6:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"
