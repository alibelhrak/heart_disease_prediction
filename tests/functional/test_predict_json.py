"""
Week 8 (Mohaned) — functional tests for the new single-patient JSON endpoint
POST /predict (request validated by api/schemas.py:PatientIn).

Equivalence classes
  valid:    required fields only; all fields; categories in any case
  invalid:  missing required field; unknown category; wrong type;
            unexpected extra field; numeric value outside its range
Boundary values: for every numeric field, min, max (accepted) and
min-0.1 / max+0.1 (rejected with 422).
"""
import pytest

from src import features as F
from tests.conftest import csv_upload

pytestmark = pytest.mark.slow

MINIMAL = {"age": 55, "sex": "Male", "cp": "asymptomatic"}
FULL = {"age": 63, "sex": "Female", "cp": "typical angina", "dataset": "Hungarian",
        "trestbps": 145.0, "chol": 233.0, "fbs": True, "restecg": "lv hypertrophy",
        "thalch": 150.0, "exang": False, "oldpeak": 2.3, "slope": "downsloping",
        "ca": 0, "thal": "fixed defect"}


def post(client, body, **params):
    return client.post("/predict", json=body, params=params)


# ── valid classes ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("body", [MINIMAL, FULL], ids=["minimal", "full"])
def test_valid_patient(api_client, body):
    r = post(api_client, body)
    assert r.status_code == 200
    out = r.json()
    assert out["prediction"] in (0, 1)
    assert 0 <= out["probability_disease"] <= 1
    assert out["probability_disease"] + out["probability_no_disease"] == pytest.approx(1)
    assert out["risk_level"] in {"Low", "Medium", "High"}
    assert out["model_used"] == api_client.get("/").json()["champion"]


def test_categories_are_case_insensitive(api_client):
    a = post(api_client, {"age": 55, "sex": "male", "cp": " ASYMPTOMATIC "}).json()
    b = post(api_client, MINIMAL).json()
    assert a["probability_disease"] == b["probability_disease"]
    assert a["input"]["sex"] == "Male"


@pytest.mark.parametrize("model", ["LogisticRegression", "RandomForest", "XGBoost"])
def test_model_selection(api_client, model):
    assert post(api_client, FULL, model_name=model).json()["model_used"] == model


def test_unknown_model_is_400(api_client):
    r = post(api_client, FULL, model_name="SVM")
    assert r.status_code == 400 and "Unknown model" in r.json()["detail"]


def test_json_and_csv_endpoints_agree(api_client):
    """Same patient through both endpoints → same probability."""
    import pandas as pd
    json_out = post(api_client, FULL).json()
    csv_out = api_client.post("/predict/csv", files=csv_upload(pd.DataFrame([FULL]))).json()
    assert csv_out["results"][0]["probability_disease"] == json_out["probability_disease"]


def test_optional_fields_use_the_csv_defaults(api_client):
    explicit = {**MINIMAL, "dataset": "Cleveland", "trestbps": 130, "chol": 240,
                "thalch": 150, "oldpeak": 0.0, "ca": 0, "restecg": "normal",
                "slope": "flat", "thal": "normal", "fbs": False, "exang": False}
    assert (post(api_client, MINIMAL).json()["probability_disease"]
            == post(api_client, explicit).json()["probability_disease"])


# ── invalid classes ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("field", ["age", "sex", "cp"])
def test_missing_required_field(api_client, field):
    body = {k: v for k, v in MINIMAL.items() if k != field}
    r = post(api_client, body)
    assert r.status_code == 422
    assert any(e["loc"][-1] == field for e in r.json()["detail"])


@pytest.mark.parametrize("field, value", [("sex", "Other"), ("cp", "chest ache"),
                                          ("thal", "unknown"), ("slope", "up"),
                                          ("restecg", 1), ("dataset", "Mars")])
def test_unknown_category(api_client, field, value):
    assert post(api_client, {**MINIMAL, field: value}).status_code == 422


@pytest.mark.parametrize("field, value", [("age", "fifty"), ("chol", "high"),
                                          ("fbs", "maybe"), ("ca", 1.5)])
def test_wrong_type(api_client, field, value):
    assert post(api_client, {**MINIMAL, field: value}).status_code == 422


def test_extra_field_rejected(api_client):
    assert post(api_client, {**MINIMAL, "weight": 80}).status_code == 422


def test_empty_body(api_client):
    assert api_client.post("/predict", json={}).status_code == 422


# ── boundary values for every numeric field ─────────────────────────────────
BOUNDS = []
for name, (lo, hi) in F.NUMERIC_RANGES.items():
    step = 1 if name in ("age", "ca") else 0.1
    BOUNDS += [(name, lo, 200), (name, hi, 200),
               (name, round(lo - step, 1), 422), (name, round(hi + step, 1), 422)]


@pytest.mark.parametrize("field, value, status", BOUNDS,
                         ids=[f"{f}={v}" for f, v, _ in BOUNDS])
def test_numeric_boundaries(api_client, field, value, status):
    body = {**MINIMAL, field: value}
    if field == "ca" and status == 422:
        body[field] = int(value)          # stay an integer so only the range fails
    assert post(api_client, body).status_code == status


def test_explicit_null_optional_fields_are_accepted(api_client):
    body = {**MINIMAL, "restecg": None, "chol": None, "fbs": None}
    assert post(api_client, body).json()["probability_disease"] == \
        post(api_client, MINIMAL).json()["probability_disease"]
