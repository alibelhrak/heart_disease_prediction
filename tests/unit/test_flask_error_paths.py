"""Week 5 (Mohaned) — Flask /predict error paths, FastAPI stubbed with requests-mock."""
import io

import pytest
import requests

from tests.conftest import upload


def _error_text(resp):
    return resp.get_data(as_text=True)


def test_no_file_field(flask_client):
    resp = flask_client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "No file uploaded." in _error_text(resp)


def test_empty_filename(flask_client):
    resp = flask_client.post("/predict", data={"file": (io.BytesIO(b""), "")},
                             content_type="multipart/form-data")
    assert "Please select a CSV file." in _error_text(resp)


@pytest.mark.parametrize("name", ["data.txt", "data.xlsx", "csv"])
def test_non_csv_rejected(flask_client, name):
    assert "Only CSV files are accepted." in _error_text(upload(flask_client, filename=name))


def test_uppercase_extension_accepted(flask_client, fastapi_stub):
    assert upload(flask_client, filename="DATA.CSV").status_code == 302


def test_backend_unreachable(flask_client, flask_app, requests_mock):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv",
                       exc=requests.exceptions.ConnectionError)
    assert "Cannot reach FastAPI backend" in _error_text(upload(flask_client))


def test_backend_timeout(flask_client, flask_app, requests_mock):
    # ReadTimeout, not ConnectTimeout: the latter is also a ConnectionError
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv",
                       exc=requests.exceptions.ReadTimeout)
    assert "Request timed out" in _error_text(upload(flask_client))


@pytest.mark.parametrize("status", [400, 500])
def test_backend_http_error(flask_client, flask_app, requests_mock, status):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv",
                       status_code=status, json={"detail": "boom"})
    assert "Prediction failed" in _error_text(upload(flask_client))


def test_backend_invalid_json(flask_client, flask_app, requests_mock):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv", text="not json")
    assert "Prediction failed" in _error_text(upload(flask_client))


def test_error_paths_store_nothing(flask_client, flask_app, requests_mock):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv",
                       exc=requests.exceptions.ConnectionError)
    upload(flask_client)
    assert len(flask_app.RESULT_STORE) == 0


def test_success_forwards_file_and_model(flask_client, fastapi_stub):
    resp = upload(flask_client, content=b"age\n61\n", model="XGBoost")
    assert resp.status_code == 302 and resp.location.endswith("/results")
    sent = fastapi_stub.last_request
    assert b"61" in sent.body and b"XGBoost" in sent.body



# Week 9 (Mohaned) — the backend's reason is shown, not just the status line
@pytest.mark.parametrize("body, shown", [
    ({"detail": "Unknown model 'SVM'"}, "Unknown model"),
    ({"detail": [{"msg": "Input should be greater than or equal to 1"}]},
     "greater than or equal to 1"),
    ({"detail": ["plain"]}, "plain"),
    ({"other": "x"}, "HTTP 400"),
    ([1, 2], "HTTP 400"),
])
def test_backend_error_detail_is_shown(flask_client, flask_app, requests_mock, body, shown):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv", status_code=400, json=body)
    html = _error_text(upload(flask_client))
    assert "Prediction failed" in html and shown in html.replace("&#39;", "'")


def test_backend_error_non_json(flask_client, flask_app, requests_mock):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv", status_code=502,
                       text="Bad Gateway", reason="Bad Gateway")
    assert "HTTP 502 Bad Gateway" in _error_text(upload(flask_client))
