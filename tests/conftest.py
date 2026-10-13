"""
Shared fixtures.

Model artifacts are never read from the repository: a small synthetic
dataset is pushed through the real preprocess -> train pipeline inside a
temporary directory once per test session, and every test that needs a
scaler, feature list or model reads them from there.
"""
import contextlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402

RAW_COLUMNS = ["id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
               "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Tiny grids so the full training stage runs in seconds
TINY_PARAMS = {
    "data": {"test_size": 0.25, "random_state": 0, "target_column": "num"},
    "logistic_regression": {"C": [1.0], "max_iter": 200},
    "random_forest": {"n_estimators": [10], "max_depth": [3], "min_samples_split": [2]},
    "xgboost": {"n_estimators": [10], "max_depth": [2],
                "learning_rate": [0.1], "subsample": [1.0]},
    "evaluation": {"primary_metric": "f1", "cv_folds": 2, "min_f1_threshold": 0.0},
}


# ── Synthetic data ──────────────────────────────────────────────────────────
def make_raw_df(n: int = 120, seed: int = 0, missing: bool = True) -> pd.DataFrame:
    """Raw rows in the UCI schema. Every category appears at least once."""
    rng = np.random.default_rng(seed)

    def cat(values):
        col = list(values) * (n // len(values) + 1)
        return rng.permutation(col[:n])

    age = rng.integers(29, 78, n)
    df = pd.DataFrame({
        "id":       np.arange(1, n + 1),
        "age":      age,
        "sex":      cat(F.SEX_VALS),
        "dataset":  cat(F.DATASET_VALS),
        "cp":       cat(F.CP_VALS),
        "trestbps": rng.integers(95, 190, n).astype(float),
        "chol":     rng.integers(150, 400, n).astype(float),
        "fbs":      rng.choice([True, False], n),
        "restecg":  cat(F.RESTECG_VALS),
        "thalch":   rng.integers(80, 200, n).astype(float),
        "exang":    rng.choice([True, False], n),
        "oldpeak":  np.round(rng.uniform(0, 4, n), 1),
        "slope":    cat(F.SLOPE_VALS),
        "ca":       rng.integers(0, 4, n).astype(float),
        "thal":     cat(F.THAL_VALS),
    })
    # Make the label depend on the features so the models learn something
    score = (age - 50) / 10 + df["oldpeak"] - (df["thalch"] - 140) / 30
    df["num"] = np.where(score + rng.normal(0, 0.5, n) > 0,
                         rng.integers(1, 5, n), 0)
    if missing:
        df.loc[df.index[:5], "ca"] = np.nan
        df.loc[df.index[3:8], "thal"] = np.nan
        df["fbs"] = df["fbs"].astype(object)
        df.loc[df.index[10:12], "fbs"] = np.nan
    return df[RAW_COLUMNS]


@pytest.fixture
def raw_df():
    return make_raw_df()


# ── Real pipeline run into a temp directory (once per session) ─────────────
@pytest.fixture(scope="session")
def pipeline_artifacts(tmp_path_factory):
    from src import preprocess, train

    base = tmp_path_factory.mktemp("pipeline")
    raw_csv = base / "raw.csv"
    proc_dir, model_dir = base / "processed", base / "models"
    make_raw_df(n=160, seed=1).to_csv(raw_csv, index=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(base)                       # MLflow writes its artifacts here
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(base / 'mlflow.db').as_posix()}")
        mp.setattr(preprocess, "RAW", raw_csv)
        mp.setattr(preprocess, "PROC", proc_dir)
        mp.setattr(train, "PROC", proc_dir)
        mp.setattr(train, "MODEL_DIR", model_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            preprocess.preprocess(TINY_PARAMS)
            train.train(TINY_PARAMS)

    return SimpleNamespace(base=base, raw_csv=raw_csv,
                           proc_dir=proc_dir, model_dir=model_dir)


# ── FastAPI ─────────────────────────────────────────────────────────────────
@pytest.fixture
def api_module(pipeline_artifacts, monkeypatch):
    import api.main as api
    monkeypatch.setattr(api, "PROC_DIR", pipeline_artifacts.proc_dir)
    monkeypatch.setattr(api, "MODEL_DIR", pipeline_artifacts.model_dir)
    return api


@pytest.fixture
def api_client(api_module):
    from fastapi.testclient import TestClient
    with TestClient(api_module.app) as client:   # runs the lifespan (load_artifacts)
        yield client


def csv_upload(df: pd.DataFrame, name: str = "patients.csv"):
    return {"file": (name, df.to_csv(index=False).encode(), "text/csv")}


# ── Flask ───────────────────────────────────────────────────────────────────
@pytest.fixture
def flask_app():
    import frontend.app as fe
    fe.RESULT_STORE.clear()
    fe.app.config.update(TESTING=True)
    yield fe
    fe.RESULT_STORE.clear()


@pytest.fixture
def flask_client(flask_app):
    return flask_app.app.test_client()


def fake_prediction_payload(n: int = 3, model: str = "LogisticRegression") -> dict:
    rows = [{
        "row_index": i, "id": i + 1, "age": 50 + i, "sex": "Male",
        "dataset": "Cleveland", "cp": "asymptomatic", "trestbps": 130.0,
        "chol": 240.0, "thalch": 150.0, "oldpeak": 1.0,
        "prediction": i % 2, "probability_disease": 0.8 if i % 2 else 0.2,
        "probability_no_disease": 0.2 if i % 2 else 0.8,
        "risk_level": "High" if i % 2 else "Low",
        "interpretation": "text, with a comma", "model_used": model,
    } for i in range(n)]
    disease = sum(r["prediction"] for r in rows)
    return {"model_used": model, "total_patients": n, "disease_count": disease,
            "no_disease_count": n - disease, "high_risk_count": disease,
            "model_metrics": {"f1": 0.8}, "results": rows, "errors": []}


@pytest.fixture
def fastapi_stub(requests_mock, flask_app):
    """Stub the FastAPI backend that the Flask frontend calls."""
    base = flask_app.FASTAPI_URL
    requests_mock.post(f"{base}/predict/csv", json=fake_prediction_payload())
    requests_mock.get(f"{base}/models", json={"available": ["LogisticRegression"],
                                              "champion": "LogisticRegression"})
    requests_mock.get(f"{base}/", json={"status": "ok"})
    return requests_mock


def upload(client, content: bytes = b"age\n50\n", filename: str = "p.csv",
           model: str = "LogisticRegression"):
    return client.post("/predict", data={"file": (io.BytesIO(content), filename),
                                         "model": model},
                       content_type="multipart/form-data")
