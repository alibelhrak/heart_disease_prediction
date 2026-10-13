"""Week 7 (Mohaned) — close branch-coverage gaps in api/main.py (endpoints)."""
import json
import shutil

import pytest

from tests.conftest import csv_upload, make_raw_df

pytestmark = pytest.mark.slow


def test_health(api_client):
    body = api_client.get("/").json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"LogisticRegression", "RandomForest", "XGBoost"}


def test_models_endpoint(api_client, pipeline_artifacts):
    body = api_client.get("/models").json()
    info = json.loads((pipeline_artifacts.model_dir / "champion_info.json").read_text())
    assert body["champion"] == info["champion"]
    assert set(body["all_metrics"]) == set(info["all_models"])


def test_predict_csv_champion(api_client):
    df = make_raw_df(n=10, seed=5)
    body = api_client.post("/predict/csv", files=csv_upload(df)).json()
    assert body["total_patients"] == 10 and body["errors"] == []
    assert body["model_used"] == api_client.get("/").json()["champion"]
    assert body["disease_count"] + body["no_disease_count"] == 10
    r = body["results"][0]
    assert 0 <= r["probability_disease"] <= 1
    assert r["probability_disease"] + r["probability_no_disease"] == pytest.approx(1)
    assert r["risk_level"] in {"Low", "Medium", "High"}


@pytest.mark.parametrize("model", ["LogisticRegression", "RandomForest", "XGBoost"])
def test_predict_csv_named_model(api_client, model):
    df = make_raw_df(n=4, seed=6)
    body = api_client.post("/predict/csv", files=csv_upload(df),
                           data={"model_name": model}).json()
    assert body["model_used"] == model
    assert all(r["model_used"] == model for r in body["results"])
    assert body["model_metrics"]          # metrics found for the chosen model


def test_predict_csv_unknown_model(api_client):
    resp = api_client.post("/predict/csv", files=csv_upload(make_raw_df(n=2)),
                           data={"model_name": "SVM"})
    assert resp.status_code == 400 and "Unknown model" in resp.json()["detail"]


def test_predict_csv_unparseable_file(api_client):
    resp = api_client.post("/predict/csv",
                           files={"file": ("empty.csv", b"", "text/csv")})
    assert resp.status_code == 400 and "Could not parse CSV" in resp.json()["detail"]


def test_predict_csv_bad_row_goes_to_errors(api_client):
    df = make_raw_df(n=3, seed=7, missing=False)
    df["id"] = df["id"].astype(object)
    df.loc[1, "id"] = "not-a-number"          # int(id) fails for this row only
    body = api_client.post("/predict/csv", files=csv_upload(df)).json()
    assert body["total_patients"] == 2
    assert [e["row"] for e in body["errors"]] == [1]


def test_predict_csv_without_id_column_uses_row_number(api_client):
    df = make_raw_df(n=3, seed=8).drop(columns=["id", "num"])
    body = api_client.post("/predict/csv", files=csv_upload(df)).json()
    assert [r["id"] for r in body["results"]] == [1, 2, 3]


def test_predict_csv_minimal_columns(api_client):
    # Only age given; every other field falls back to its default
    body = api_client.post("/predict/csv",
                           files={"file": ("a.csv", b"age\n45\n70\n", "text/csv")}).json()
    assert body["total_patients"] == 2 and body["errors"] == []


def test_missing_model_file_is_skipped(api_module, pipeline_artifacts, tmp_path,
                                       monkeypatch, capsys):
    model_dir = tmp_path / "models"
    shutil.copytree(pipeline_artifacts.model_dir, model_dir)
    (model_dir / "xgboost_model.joblib").unlink()
    monkeypatch.setattr(api_module, "MODEL_DIR", model_dir)
    api_module.load_artifacts()
    assert set(api_module.MODELS) == {"LogisticRegression", "RandomForest"}
    assert "WARNING" in capsys.readouterr().out


def test_reload_does_not_keep_stale_models(api_module, pipeline_artifacts, tmp_path,
                                           monkeypatch):
    api_module.load_artifacts()
    model_dir = tmp_path / "models"
    shutil.copytree(pipeline_artifacts.model_dir, model_dir)
    (model_dir / "randomforest_model.joblib").unlink()
    monkeypatch.setattr(api_module, "MODEL_DIR", model_dir)
    api_module.load_artifacts()
    assert "RandomForest" not in api_module.MODELS
