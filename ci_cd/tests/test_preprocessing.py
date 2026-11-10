"""Pruebas unitarias de app/preprocessing.py: funciones puras, sin HTTP ni modelo."""

import pytest

from app.preprocessing import (
    FEATURE_ORDER,
    UNKNOWN,
    age_bracket_to_years,
    build_features,
    clean_category,
    probability_to_risk,
    validate_ranges,
)

# --------------------------------------------------------------- clean_category


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Caucasian", "Caucasian"),
        ("  Caucasian ", "Caucasian"),  # espacios al borde
        ("African   American", "African American"),  # espacios internos repetidos
        ("?", UNKNOWN),  # asi viene el faltante en el CSV
        ("Unknown/Invalid", UNKNOWN),  # asi viene en gender
        ("", UNKNOWN),
        (None, UNKNOWN),
    ],
)
def test_clean_category(raw, expected):
    assert clean_category(raw) == expected


# ------------------------------------------------------------ age_bracket_to_years


@pytest.mark.parametrize(
    "bracket, expected",
    [("[0-10)", 5), ("[70-80)", 75), ("[90-100)", 95), (" [40-50) ", 45)],
)
def test_age_bracket_to_years(bracket, expected):
    assert age_bracket_to_years(bracket) == expected


@pytest.mark.parametrize("bad", ["70-80", "[70-80]", "[80-70)", "[90-110)", "setenta", ""])
def test_age_bracket_to_years_invalid_raises(bad):
    with pytest.raises(ValueError, match="rango de edad"):
        age_bracket_to_years(bad)


# ---------------------------------------------------------------- validate_ranges


def test_validate_ranges_returns_ints(valid_record):
    # Arrange: el JSON puede traer 3.0 en lugar de 3
    valid_record["time_in_hospital"] = 3.0
    # Act
    result = validate_ranges(valid_record)
    # Assert
    assert result["time_in_hospital"] == 3
    assert all(isinstance(v, int) for v in result.values())


def test_validate_ranges_ignores_non_numeric_columns(valid_record):
    assert "race" not in validate_ranges(valid_record)


def test_validate_ranges_missing_column_raises(valid_record):
    del valid_record["num_medications"]
    with pytest.raises(KeyError, match="num_medications"):
        validate_ranges(valid_record)


@pytest.mark.parametrize("column, value", [("time_in_hospital", 1), ("time_in_hospital", 14)])
def test_validate_ranges_accepts_exact_borders(valid_record, column, value):
    valid_record[column] = value
    assert validate_ranges(valid_record)[column] == value


@pytest.mark.parametrize(
    "column, value",
    [
        ("time_in_hospital", 0),  # justo debajo del minimo
        ("time_in_hospital", 15),  # justo encima del maximo
        ("number_diagnoses", 0),
        ("num_medications", -1),  # negativo imposible
    ],
)
def test_validate_ranges_out_of_range_raises(valid_record, column, value):
    valid_record[column] = value
    with pytest.raises(ValueError, match=f"{column}.*fuera de rango"):
        validate_ranges(valid_record)


@pytest.mark.parametrize(
    "value, message",
    [("tres", "numerico"), (None, "numerico"), (True, "numerico"), (2.5, "entero")],
)
def test_validate_ranges_wrong_type_raises(valid_record, value, message):
    valid_record["num_procedures"] = value
    with pytest.raises(ValueError, match=message):
        validate_ranges(valid_record)


# ----------------------------------------------------------------- build_features


def test_build_features_keeps_model_column_order(valid_record):
    """Si alguien reordena FEATURE_ORDER o el dict, esta prueba lo detecta."""
    assert list(build_features(valid_record)) == FEATURE_ORDER


def test_build_features_transforms_and_cleans(valid_record):
    valid_record["race"] = "?"
    features = build_features(valid_record)
    assert features["age_years"] == 75
    assert features["race"] == UNKNOWN


def test_build_features_optional_categoricals_default_to_unknown(valid_record):
    del valid_record["A1Cresult"]
    assert build_features(valid_record)["A1Cresult"] == UNKNOWN


def test_build_features_missing_age_raises(valid_record):
    del valid_record["age"]
    with pytest.raises(KeyError, match="age"):
        build_features(valid_record)


# ------------------------------------------------------------ probability_to_risk


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.95, "high"),
        (0.6, "high"),  # borde exacto
        (0.59, "medium"),
        (0.3, "medium"),  # borde exacto
        (0.29, "low"),
        (0.0, "low"),
        (1.0, "high"),
    ],
)
def test_probability_to_risk(probability, expected):
    assert probability_to_risk(probability) == expected


@pytest.mark.parametrize("bad", [-0.01, 1.01, 50])
def test_probability_to_risk_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="entre 0 y 1"):
        probability_to_risk(bad)
