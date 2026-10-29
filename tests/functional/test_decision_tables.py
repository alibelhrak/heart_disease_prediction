"""
Week 8 (Mohaned) — decision-table tests.

Table 1: Flask POST /predict (FastAPI backend stubbed with requests-mock)

  Rule | file field | filename | .csv | backend         | Outcome
  -----+------------+----------+------+-----------------+-----------------------------------
  R1   | no         | -        | -    | -               | "No file uploaded."
  R2   | yes        | empty    | -    | -               | "Please select a CSV file."
  R3   | yes        | given    | no   | -               | "Only CSV files are accepted."
  R4   | yes        | given    | yes  | 200 OK          | 302 → /results, result stored
  R5   | yes        | given    | yes  | connection err  | "Cannot reach FastAPI backend"
  R6   | yes        | given    | yes  | read timeout    | "Request timed out"
  R7   | yes        | given    | yes  | HTTP 4xx/5xx    | "Prediction failed"
  R8   | yes        | given    | yes  | invalid JSON    | "Prediction failed"

  Only R4 may call store_result; R1–R3 must never call the backend.

Table 2: FastAPI POST /predict/csv

  Rule | CSV parses | model_name          | Outcome
  -----+------------+---------------------+--------------------------------
  A1   | no         | any                 | 400 "Could not parse CSV"
  A2   | yes        | "champion"/omitted  | 200, model_used = champion
  A3   | yes        | a loaded model      | 200, model_used = that model
  A4   | yes        | unknown             | 400 "Unknown model"
"""
import io

import pytest
import requests

from tests.conftest import csv_upload, fake_prediction_payload, make_raw_df

OK, CONN, TIMEOUT, HTTP500, BADJSON = "ok", "conn", "timeout", "http500", "badjson"

FLASK_TABLE = [
    # rule, has_field, filename, backend, expected text (None = redirect)
    ("R1", False, None,         None,    "No file uploaded."),
    ("R2", True,  "",           None,    "Please select a CSV file."),
    ("R3", True,  "notes.txt",  None,    "Only CSV files are accepted."),
    ("R4", True,  "p.csv",      OK,      None),
    ("R5", True,  "p.csv",      CONN,    "Cannot reach FastAPI backend"),
    ("R6", True,  "p.csv",      TIMEOUT, "Request timed out"),
    ("R7", True,  "p.csv",      HTTP500, "Prediction failed"),
    ("R8", True,  "p.csv",      BADJSON, "Prediction failed"),
]


def _arm_backend(requests_mock, url, backend):
    target = f"{url}/predict/csv"
    if backend == OK:
        requests_mock.post(target, json=fake_prediction_payload())
    elif backend == CONN:
        requests_mock.post(target, exc=requests.exceptions.ConnectionError)
    elif backend == TIMEOUT:
        requests_mock.post(target, exc=requests.exceptions.ReadTimeout)
    elif backend == HTTP500:
        requests_mock.post(target, status_code=500, json={"detail": "boom"})
    elif backend == BADJSON:
        requests_mock.post(target, text="<html>not json</html>")


@pytest.mark.parametrize("rule, has_field, filename, backend, expected", FLASK_TABLE,
                         ids=[r[0] for r in FLASK_TABLE])
def test_flask_predict_decision_table(flask_client, flask_app, requests_mock,
                                      rule, has_field, filename, backend, expected):
    _arm_backend(requests_mock, flask_app.FASTAPI_URL, backend)
    data = {"model": "LogisticRegression"}
    if has_field:
        data["file"] = (io.BytesIO(b"age\n50\n"), filename)
    resp = flask_client.post("/predict", data=data, content_type="multipart/form-data")

    if expected is None:
        assert resp.status_code == 302 and resp.location.endswith("/results")
        assert len(flask_app.RESULT_STORE) == 1
    else:
        assert resp.status_code == 200
        assert expected in resp.get_data(as_text=True)
        assert len(flask_app.RESULT_STORE) == 0
    # Validation rules (R1–R3) are decided locally, without calling the backend
    assert requests_mock.called is (backend is not None)


API_TABLE = [
    # rule, csv bytes, model_name (None = omit), expected status, expected model/detail
    ("A1", b"",   "champion",           400, "Could not parse CSV"),
    ("A1b", b"",  "SVM",                400, "Could not parse CSV"),   # parse checked first
    ("A2", None,  None,                 200, "champion"),
    ("A2b", None, "champion",           200, "champion"),
    ("A3", None,  "RandomForest",       200, "RandomForest"),
    ("A4", None,  "SVM",                400, "Unknown model"),
]


@pytest.mark.slow
@pytest.mark.parametrize("rule, content, model, status, expect", API_TABLE,
                         ids=[r[0] for r in API_TABLE])
def test_api_predict_csv_decision_table(api_client, rule, content, model, status, expect):
    files = (csv_upload(make_raw_df(n=3, seed=2)) if content is None
             else {"file": ("x.csv", content, "text/csv")})
    data = {} if model is None else {"model_name": model}
    resp = api_client.post("/predict/csv", files=files, data=data)
    assert resp.status_code == status
    if status == 200:
        champion = api_client.get("/").json()["champion"]
        assert resp.json()["model_used"] == (champion if expect == "champion" else expect)
    else:
        assert expect in resp.json()["detail"]


@pytest.mark.slow
def test_blank_id_cell_still_scored(api_client):
    """Defect found: a blank id cell used to drop the whole patient (int(NaN))."""
    csv = b"id,age,sex,cp\n,50,Male,asymptomatic\n7,60,Female,non-anginal\n"
    body = api_client.post("/predict/csv", files={"file": ("p.csv", csv, "text/csv")}).json()
    assert body["errors"] == []
    assert [r["id"] for r in body["results"]] == [1, 7]
