"""Week 7 (Mohaned) — close branch-coverage gaps in frontend/app.py."""
import requests

from frontend.app import results_to_csv


def test_index_renders_upload_page(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200
    assert 'name="file"' in resp.get_data(as_text=True)


def test_api_models_proxies_backend(flask_client, fastapi_stub):
    body = flask_client.get("/api/models").get_json()
    assert body == {"available": ["LogisticRegression"], "champion": "LogisticRegression"}


def test_api_models_falls_back_when_backend_down(flask_client, flask_app, requests_mock):
    requests_mock.get(f"{flask_app.FASTAPI_URL}/models",
                      exc=requests.exceptions.ConnectionError)
    body = flask_client.get("/api/models").get_json()
    assert set(body["available"]) == {"LogisticRegression", "RandomForest", "XGBoost"}


def test_health_backend_ok(flask_client, fastapi_stub):
    assert flask_client.get("/health").get_json() == {"flask": "ok", "fastapi": "ok"}


def test_health_backend_non_200(flask_client, flask_app, requests_mock):
    requests_mock.get(f"{flask_app.FASTAPI_URL}/", status_code=503)
    assert flask_client.get("/health").get_json()["fastapi"] == "unreachable"


def test_health_backend_down(flask_client, flask_app, requests_mock):
    requests_mock.get(f"{flask_app.FASTAPI_URL}/", exc=requests.exceptions.ConnectionError)
    assert flask_client.get("/health").get_json()["fastapi"] == "unreachable"


def test_results_to_csv_none_and_missing_values():
    out = results_to_csv([{"id": 1, "age": None}]).decode().splitlines()
    assert out[0].startswith("id,age,sex")
    assert out[1].startswith("1,,")


def test_results_to_csv_escapes_quotes():
    out = results_to_csv([{"id": 1, "interpretation": 'say "hi", ok'}]).decode()
    assert '"say ""hi"", ok"' in out
