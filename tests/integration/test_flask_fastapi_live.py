"""
Week 9 (Mohaned) — Flask ↔ FastAPI integration over real HTTP.

Weeks 5–7 tested the Flask frontend with the backend stubbed by
requests-mock. Here the stub is removed: the real FastAPI app (with models
from the tmp pipeline) runs in a uvicorn server on a free local port, and the
Flask test client talks to it with real `requests` calls. This checks the
contract between the two services: field names in the multipart request,
the JSON response shape the templates rely on, and the error paths.
"""
import csv
import io
import socket
import threading
import time

import pytest
import requests
import uvicorn

from tests.conftest import make_raw_df, upload

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api(pipeline_artifacts):
    import api.main as api
    mp = pytest.MonkeyPatch()
    mp.setattr(api, "PROC_DIR", pipeline_artifacts.proc_dir)
    mp.setattr(api, "MODEL_DIR", pipeline_artifacts.model_dir)
    # Local calls must not go through any HTTP proxy configured in the env
    for var in ("NO_PROXY", "no_proxy"):
        mp.setenv(var, "127.0.0.1,localhost")

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):                                   # wait up to 10 s
        try:
            if requests.get(f"{url}/", timeout=0.5).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        pytest.fail("FastAPI did not start")
    yield url
    server.should_exit = True
    thread.join(timeout=5)
    mp.undo()


@pytest.fixture
def client(flask_app, live_api, monkeypatch):
    monkeypatch.setattr(flask_app, "FASTAPI_URL", live_api)
    return flask_app.app.test_client()


def _csv(df) -> bytes:
    return df.to_csv(index=False).encode()


def test_health_sees_the_real_backend(client):
    assert client.get("/health").get_json() == {"flask": "ok", "fastapi": "ok"}


def test_model_list_comes_from_the_backend(client, live_api):
    body = client.get("/api/models").get_json()
    assert body == requests.get(f"{live_api}/models", timeout=5).json()


def test_upload_to_results_end_to_end(client):
    df = make_raw_df(n=12, seed=21)
    resp = upload(client, content=_csv(df), model="champion")
    assert resp.status_code == 302 and resp.location.endswith("/results")
    html = client.get("/results").get_data(as_text=True)
    assert "12 patients" in html
    assert 'id="validation-notices"' not in html


@pytest.mark.parametrize("model", ["LogisticRegression", "RandomForest", "XGBoost"])
def test_selected_model_reaches_the_backend(client, flask_app, model):
    upload(client, content=_csv(make_raw_df(n=3, seed=22)), model=model)
    stored = next(reversed(flask_app.RESULT_STORE.values()))
    assert stored["data"]["model_used"] == model


def test_export_matches_backend_predictions(client, live_api):
    df = make_raw_df(n=8, seed=23)
    upload(client, content=_csv(df), model="XGBoost")
    exported = list(csv.DictReader(io.StringIO(client.get("/export").get_data(as_text=True))))

    direct = requests.post(f"{live_api}/predict/csv",
                           files={"file": ("p.csv", _csv(df), "text/csv")},
                           data={"model_name": "XGBoost"}, timeout=30).json()["results"]
    assert len(exported) == len(direct) == 8
    for e, d in zip(exported, direct):
        assert int(e["id"]) == d["id"]
        assert float(e["probability_disease"]) == d["probability_disease"]
        assert e["risk_level"] == d["risk_level"]


def test_invalid_rows_are_shown_to_the_user(client):
    df = make_raw_df(n=4, seed=24, missing=False)
    df.loc[2, "age"] = 450
    upload(client, content=_csv(df))
    html = client.get("/results").get_data(as_text=True)
    assert "3 patients" in html
    assert "Row 3: age: 450 is outside the allowed range" in html


def test_backend_rejection_is_reported(client):
    """Defect found: Flask used to show only "400 Client Error" and drop the
    backend's explanation."""
    resp = upload(client, content=_csv(make_raw_df(n=2)), model="SVM")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Prediction failed" in html and "Unknown model" in html


def test_file_with_no_usable_columns_is_reported(client):
    resp = upload(client, content=b"name,weight\nA,80\n")
    assert "CSV has none of the expected columns" in resp.get_data(as_text=True)


def test_backend_down(flask_app, monkeypatch):
    monkeypatch.setattr(flask_app, "FASTAPI_URL", f"http://127.0.0.1:{_free_port()}")
    client = flask_app.app.test_client()
    assert "Cannot reach FastAPI backend" in upload(client).get_data(as_text=True)
    assert client.get("/health").get_json()["fastapi"] == "unreachable"
