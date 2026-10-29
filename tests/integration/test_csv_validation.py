"""
Week 9 (Mohaned) — CSV validation layer (api/validation.py) on its own and
integrated into POST /predict/csv and the Flask results page.

Before this layer, an unknown category, an impossible age or a non-numeric
cholesterol was silently replaced by a default and the patient was scored
anyway. Now the row is reported in `errors` and not scored, and missing or
ignored columns are reported in `warnings`.
"""
import pandas as pd
import pytest

from api import validation as V
from tests.conftest import csv_upload, make_raw_df


# ── unit level ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("col, val, ok", [
    ("age", 50, True), ("age", "50", True), ("age", 1, True), ("age", 120, True),
    ("age", 0, False), ("age", 121, False), ("age", "fifty", False),
    ("age", float("inf"), False), ("age", "nan", False),
    ("chol", 0, True), ("chol", 700.0, True), ("chol", 700.1, False),
    ("oldpeak", -3, True), ("oldpeak", -3.1, False),
    ("ca", 2.0, True), ("ca", 1.5, False), ("ca", 4, False),
    ("id", 7, True), ("id", 7.5, False), ("id", "x", False),
])
def test_check_number(col, val, ok):
    assert (V.check_number(col, val) is None) is ok


@pytest.mark.parametrize("col, val, ok", [
    ("sex", "Male", True), ("sex", "female", True), ("cp", " Asymptomatic ", True),
    ("sex", "Mael", False), ("thal", "normal ", True), ("thal", "3", False),
    ("dataset", "Mars", False),
])
def test_check_category(col, val, ok):
    assert (V.check_category(col, val) is None) is ok


@pytest.mark.parametrize("val, ok", [(True, True), (False, True), ("true", True), ("1", True),
                                     (0, True), (1.0, True), (" True ", True),
                                     ("maybe", False), ("yes", False), ([1], False)])
def test_check_bool(val, ok):
    assert (V.check_bool("fbs", val) is None) is ok


def test_messages_name_column_and_value():
    assert V.check_number("age", 450) == "age: 450 is outside the allowed range 1–120"
    assert "'Mael' is not one of Male, Female" in V.check_category("sex", "Mael")


def test_blank_cells_are_allowed():
    assert V.check_row({"age": 50, "chol": float("nan"), "sex": "", "fbs": None}) == []


def test_row_collects_every_problem():
    problems = V.check_row({"age": 300, "sex": "X", "fbs": "maybe", "chol": 200})
    assert [p.split(":")[0] for p in problems] == ["age", "sex", "fbs"]


def test_validate_frame_warnings():
    df = pd.DataFrame({"id": [1], "age": [50], "sex": ["Male"], "weight": [80]})
    errors, warnings = V.validate_frame(df)
    assert errors == {}
    assert any("'chol' is missing" in w for w in warnings)
    assert any("'weight' is not used" in w for w in warnings)
    assert not any("'id'" in w for w in warnings)


def test_validate_frame_complete_file_has_no_warnings():
    errors, warnings = V.validate_frame(make_raw_df(n=20))
    assert errors == {} and warnings == []


def test_validate_frame_without_feature_columns():
    with pytest.raises(V.CSVValidationError, match="none of the expected columns"):
        V.validate_frame(pd.DataFrame({"name": ["a"], "weight": [80]}))


def test_generated_dataset_passes_validation(tmp_path):
    """The project's own data generator must produce only valid rows."""
    from tests.conftest import generated_dataset
    df = generated_dataset(tmp_path)
    errors, warnings = V.validate_frame(df)
    assert len(df) == 920 and errors == {} and warnings == []


# ── integrated into the API ─────────────────────────────────────────────────
pytest_slow = pytest.mark.slow


@pytest_slow
def test_invalid_rows_are_reported_not_scored(api_client):
    df = make_raw_df(n=5, seed=3, missing=False)
    df["sex"] = df["sex"].astype(object)
    df.loc[1, "age"] = 450
    df.loc[3, "sex"] = "Mael"
    body = api_client.post("/predict/csv", files=csv_upload(df)).json()
    assert body["total_patients"] == 3
    assert [r["row_index"] for r in body["results"]] == [0, 2, 4]
    assert body["errors"] == [
        {"row": 1, "error": "age: 450 is outside the allowed range 1–120"},
        {"row": 3, "error": "sex: 'Mael' is not one of Male, Female"},
    ]


@pytest_slow
def test_missing_columns_are_warned(api_client):
    body = api_client.post("/predict/csv",
                           files={"file": ("a.csv", b"age,sex\n45,Male\n", "text/csv")}).json()
    assert body["total_patients"] == 1
    assert len(body["warnings"]) == len(V.FEATURE_COLUMNS) - 2


@pytest_slow
def test_file_without_feature_columns_is_400(api_client):
    resp = api_client.post("/predict/csv",
                           files={"file": ("a.csv", b"name,weight\nA,80\n", "text/csv")})
    assert resp.status_code == 400 and "none of the expected columns" in resp.json()["detail"]


@pytest_slow
def test_header_whitespace_is_tolerated(api_client):
    body = api_client.post("/predict/csv",
                           files={"file": ("a.csv", b" age , sex \n45,Male\n", "text/csv")}).json()
    assert body["total_patients"] == 1 and body["errors"] == []


@pytest_slow
def test_all_rows_invalid(api_client):
    body = api_client.post("/predict/csv",
                           files={"file": ("a.csv", b"age\n0\n500\n", "text/csv")}).json()
    assert body["total_patients"] == 0 and len(body["errors"]) == 2


# ── shown to the user by the Flask results page ─────────────────────────────
def test_results_page_lists_skipped_rows_and_warnings(flask_client, flask_app, requests_mock):
    from tests.conftest import fake_prediction_payload, upload
    payload = fake_prediction_payload(n=2)
    payload["errors"] = [{"row": 4, "error": "age: 450 is outside the allowed range 1–120"}]
    payload["warnings"] = ["Column 'chol' is missing; the default 240 is used for every patient."]
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv", json=payload)
    upload(flask_client)
    html = flask_client.get("/results").get_data(as_text=True)
    assert "1 row skipped" in html
    assert "Row 5: age: 450 is outside the allowed range" in html
    assert "Column &#39;chol&#39; is missing" in html


def test_results_page_truncates_long_error_lists(flask_client, flask_app, requests_mock):
    from tests.conftest import fake_prediction_payload, upload
    payload = fake_prediction_payload(n=1)
    payload["errors"] = [{"row": i, "error": "age: bad"} for i in range(25)]
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv", json=payload)
    upload(flask_client)
    html = flask_client.get("/results").get_data(as_text=True)
    assert "25 rows skipped" in html and "… and 5 more" in html


def test_results_page_without_notices(flask_client, fastapi_stub):
    from tests.conftest import upload
    upload(flask_client)
    assert 'id="validation-notices"' not in flask_client.get("/results").get_data(as_text=True)


@pytest_slow
def test_unexpected_scoring_failure_is_isolated_to_its_row(api_client, api_module, monkeypatch):
    """Defence in depth: a row that passes validation but still fails while
    being scored is reported in `errors` and does not abort the file."""
    real_score = api_module.score
    calls = {"n": 0}

    def flaky(model, X):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("model blew up")
        return real_score(model, X)

    monkeypatch.setattr(api_module, "score", flaky)
    body = api_client.post("/predict/csv", files=csv_upload(make_raw_df(n=3, seed=9))).json()
    assert body["total_patients"] == 2
    assert body["errors"] == [{"row": 1, "error": "model blew up"}]
