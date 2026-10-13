"""
Week 6 (Mohaned) — def-use tests for RESULT_STORE across /predict → /results → /export.

  Definition (def)                            Uses
  ------------------------------------------  -----------------------------------------
  /predict: store_result() writes             /results: RESULT_STORE[id] -> template
  RESULT_STORE[id] and session["result_id"]   /export:  RESULT_STORE[id]["data"]["results"]
  (kill: next /predict in the same session,          -> CSV body
   or eviction when MAX_STORED_RESULTS is hit)

Also verifies the two leak fixes: the store is bounded, and /export writes
no temporary files.
"""
import csv
import io
import os
import tempfile

from tests.conftest import fake_prediction_payload, upload


# ── def → use pairs ─────────────────────────────────────────────────────────
def test_predict_defines_entry_and_session_key(flask_client, flask_app, fastapi_stub):
    upload(flask_client, filename="cohort.csv", model="XGBoost")
    assert len(flask_app.RESULT_STORE) == 1
    (result_id, entry), = flask_app.RESULT_STORE.items()
    with flask_client.session_transaction() as sess:
        assert sess["result_id"] == result_id
    assert entry["filename"] == "cohort.csv"
    assert entry["model"] == "XGBoost"
    assert entry["data"] == fake_prediction_payload()


def test_def_predict_use_results(flask_client, fastapi_stub):
    upload(flask_client, filename="cohort.csv")
    resp = flask_client.get("/results")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "cohort.csv" in html
    assert "LogisticRegression" in html


def test_def_predict_use_export(flask_client, fastapi_stub):
    upload(flask_client)
    resp = flask_client.get("/export")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert "heart_disease_predictions.csv" in resp.headers["Content-Disposition"]
    rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
    expected = fake_prediction_payload()["results"]
    assert len(rows) == len(expected)
    assert [r["id"] for r in rows] == [str(e["id"]) for e in expected]
    # value containing a comma survives the round trip intact
    assert rows[0]["interpretation"] == "text, with a comma"


def test_def_predict_use_results_then_export(flask_client, fastapi_stub):
    upload(flask_client)
    assert flask_client.get("/results").status_code == 200
    assert flask_client.get("/export").status_code == 200   # entry not consumed


# ── uses with no reaching definition ────────────────────────────────────────
def test_results_without_session_redirects(flask_client):
    resp = flask_client.get("/results")
    assert resp.status_code == 302 and resp.location.endswith("/")


def test_export_without_session_redirects(flask_client):
    assert flask_client.get("/export").status_code == 302


def test_stale_session_id_redirects(flask_client):
    with flask_client.session_transaction() as sess:
        sess["result_id"] = "does-not-exist"
    assert flask_client.get("/results").status_code == 302
    assert flask_client.get("/export").status_code == 302


def test_export_with_empty_results_redirects_to_results(flask_client, flask_app,
                                                        requests_mock):
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv",
                       json=fake_prediction_payload(n=0))
    upload(flask_client)
    resp = flask_client.get("/export")
    assert resp.status_code == 302 and resp.location.endswith("/results")


def test_results_page_with_zero_patients_does_not_crash(flask_client, flask_app,
                                                        requests_mock):
    # Every row failed in the API → total_patients == 0 (was a ZeroDivisionError)
    requests_mock.post(f"{flask_app.FASTAPI_URL}/predict/csv",
                       json=fake_prediction_payload(n=0))
    upload(flask_client)
    assert flask_client.get("/results").status_code == 200


# ── kill: redefinition and eviction (memory-leak fix) ──────────────────────
def test_second_predict_replaces_first_in_same_session(flask_client, flask_app,
                                                       fastapi_stub):
    upload(flask_client, filename="first.csv")
    upload(flask_client, filename="second.csv")
    assert len(flask_app.RESULT_STORE) == 1
    assert "second.csv" in flask_client.get("/results").get_data(as_text=True)


def test_store_is_bounded_across_sessions(flask_app, fastapi_stub, monkeypatch):
    monkeypatch.setattr(flask_app, "MAX_STORED_RESULTS", 3)
    clients = [flask_app.app.test_client() for _ in range(5)]
    for c in clients:
        upload(c)
    assert len(flask_app.RESULT_STORE) == 3
    # the two oldest sessions were evicted, the newest still work
    assert clients[0].get("/results").status_code == 302
    assert clients[1].get("/results").status_code == 302
    assert clients[-1].get("/results").status_code == 200


def test_sessions_are_isolated(flask_app, requests_mock):
    base = flask_app.FASTAPI_URL
    a, b = flask_app.app.test_client(), flask_app.app.test_client()
    requests_mock.post(f"{base}/predict/csv", json=fake_prediction_payload(n=2))
    upload(a, filename="a.csv")
    requests_mock.post(f"{base}/predict/csv", json=fake_prediction_payload(n=4))
    upload(b, filename="b.csv")
    assert "a.csv" in a.get("/results").get_data(as_text=True)
    assert "b.csv" in b.get("/results").get_data(as_text=True)


# ── temp-file leak fix ──────────────────────────────────────────────────────
def test_export_creates_no_temp_files(flask_client, fastapi_stub, tmp_path,
                                      monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    upload(flask_client)
    for _ in range(3):
        assert flask_client.get("/export").status_code == 200
    assert os.listdir(tmp_path) == []
