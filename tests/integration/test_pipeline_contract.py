"""
Week 9 (Ali) — contract tests between the stages
preprocess  →  train  →  API.

Each stage writes files the next stage reads. These tests pin down that
interface: which files exist, their schema, and that what one stage writes
is exactly what the next stage (and dvc.yaml) expects.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src import train as T
from tests.conftest import ROOT, TINY_PARAMS

pytestmark = pytest.mark.slow

METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}
MODEL_NAMES = {"LogisticRegression", "RandomForest", "XGBoost"}


@pytest.fixture(scope="module")
def dvc():
    return yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]


@pytest.fixture(scope="module")
def info(pipeline_artifacts):
    return json.loads((pipeline_artifacts.model_dir / "champion_info.json").read_text())


# ── preprocess → train ──────────────────────────────────────────────────────
def test_preprocess_writes_exactly_the_dvc_outs(pipeline_artifacts, dvc):
    declared = {Path(o).name for o in dvc["preprocess"]["outs"]}
    written = {p.name for p in pipeline_artifacts.proc_dir.iterdir()}
    assert written == declared


def test_train_reads_only_what_preprocess_writes(dvc):
    preprocess_outs = set(dvc["preprocess"]["outs"])
    data_deps = {d for d in dvc["train"]["deps"] if d.startswith("data/")}
    assert data_deps <= preprocess_outs


def test_train_writes_exactly_the_dvc_outs(pipeline_artifacts, dvc):
    declared = {Path(o).name for o in dvc["train"]["outs"]}
    declared |= {Path(next(iter(m))).name for m in dvc["train"]["metrics"]}
    written = {p.name for p in pipeline_artifacts.model_dir.iterdir()}
    assert written == declared


def test_dvc_params_sections_exist_in_params_yaml(dvc):
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    for stage in dvc.values():
        for section in stage.get("params", []):
            assert section in params


def test_split_files_schema(pipeline_artifacts):
    d = pipeline_artifacts.proc_dir
    X_train, X_test = pd.read_csv(d / "X_train.csv"), pd.read_csv(d / "X_test.csv")
    y_train = pd.read_csv(d / "y_train.csv")
    y_test = pd.read_csv(d / "y_test.csv")
    features = json.loads((d / "feature_names.json").read_text())

    assert list(X_train.columns) == list(X_test.columns) == features
    assert y_train.shape[1] == 1 and y_test.shape[1] == 1
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert set(y_train.iloc[:, 0]) | set(y_test.iloc[:, 0]) <= {0, 1}
    assert not X_train.isna().any().any() and not X_test.isna().any().any()


def test_split_size_and_stratification(pipeline_artifacts):
    d = pipeline_artifacts.proc_dir
    y_train = pd.read_csv(d / "y_train.csv").squeeze()
    y_test = pd.read_csv(d / "y_test.csv").squeeze()
    n = len(y_train) + len(y_test)
    assert len(y_test) == int(np.ceil(n * TINY_PARAMS["data"]["test_size"]))
    # Stratified: prevalence differs by less than one patient's worth
    assert abs(y_train.mean() - y_test.mean()) < 1 / len(y_test)


def test_train_consumes_the_feature_order(pipeline_artifacts, info):
    features = json.loads((pipeline_artifacts.proc_dir / "feature_names.json").read_text())
    assert info["feature_names"] == features
    for name in MODEL_NAMES:
        model = joblib.load(pipeline_artifacts.model_dir / f"{name.lower()}_model.joblib")
        assert model.n_features_in_ == len(features)


# ── train → API (champion_info.json is the contract) ────────────────────────
def test_champion_info_schema(info):
    assert set(info) == {"champion", "run_id", "metrics", "all_models", "feature_names"}
    assert set(info["all_models"]) == MODEL_NAMES
    assert info["champion"] in MODEL_NAMES
    for entry in info["all_models"].values():
        assert set(entry) == {"metrics", "run_id", "path"}
        assert set(entry["metrics"]) == METRIC_KEYS
        assert all(0.0 <= v <= 1.0 for v in entry["metrics"].values())


def test_model_paths_are_portable(info, pipeline_artifacts):
    """The committed champion_info.json once held absolute Windows paths
    (C:\\Users\\...); paths must be bare file names next to the JSON."""
    for entry in info["all_models"].values():
        assert Path(entry["path"]).name == entry["path"]
        assert (pipeline_artifacts.model_dir / entry["path"]).exists()


def test_champion_is_the_best_by_primary_metric(info):
    best = max(info["all_models"], key=lambda n: info["all_models"][n]["metrics"]["f1"])
    assert info["champion"] == best
    assert info["metrics"] == info["all_models"][best]["metrics"]
    assert info["run_id"] == info["all_models"][best]["run_id"]


def test_champion_model_file_is_the_champion(pipeline_artifacts, info):
    d = pipeline_artifacts.model_dir
    X = pd.read_csv(pipeline_artifacts.proc_dir / "X_test.csv")
    champ = joblib.load(d / "champion_model.joblib")
    named = joblib.load(d / info["all_models"][info["champion"]]["path"])
    np.testing.assert_array_equal(champ.predict_proba(X), named.predict_proba(X))


def test_api_serves_what_train_wrote(api_client, info):
    models = api_client.get("/models").json()
    assert set(models["available"]) == MODEL_NAMES
    assert models["champion"] == info["champion"]
    assert models["all_metrics"] == {n: e["metrics"] for n, e in info["all_models"].items()}


def test_api_cannot_start_without_artifacts(api_module, tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "PROC_DIR", tmp_path)
    monkeypatch.setattr(api_module, "MODEL_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        api_module.load_artifacts()


def test_metrics_in_champion_info_are_reproducible(pipeline_artifacts, info):
    """Re-scoring the saved models on the saved test split gives the logged metrics."""
    d = pipeline_artifacts.proc_dir
    X = pd.read_csv(d / "X_test.csv")
    y = pd.read_csv(d / "y_test.csv").squeeze()
    for name, entry in info["all_models"].items():
        model = joblib.load(pipeline_artifacts.model_dir / entry["path"])
        got = T.compute_metrics(y, model.predict(X), model.predict_proba(X)[:, 1])
        assert got == entry["metrics"], name
