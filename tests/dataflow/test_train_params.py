"""
Week 6 (Ali) — params.yaml (def) → train() (uses).

The hyperparameter grids, CV folds, scoring metric and seed must come from
params.yaml rather than being hard-coded in src/train.py.
"""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src import train as T
from tests.conftest import TINY_PARAMS


def test_load_params_reads_repository_file():
    params = T.load_params()
    on_disk = yaml.safe_load(T.PARAMS_FILE.read_text())
    assert params == on_disk
    for section in ("logistic_regression", "random_forest", "xgboost", "evaluation"):
        assert section in params


def test_load_params_custom_path(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(TINY_PARAMS))
    assert T.load_params(f) == TINY_PARAMS


def test_build_grids_uses_params_values():
    grids = T.build_grids(TINY_PARAMS, spw=2.0, random_state=7)
    lr_est, lr_grid = grids["LogisticRegression"]
    assert lr_grid == {"C": [1.0]}
    assert lr_est.max_iter == 200 and lr_est.random_state == 7
    assert grids["RandomForest"][1] == {"n_estimators": [10], "max_depth": [3],
                                        "min_samples_split": [2]}
    xgb_est, xgb_grid = grids["XGBoost"]
    assert xgb_grid["learning_rate"] == [0.1]
    assert xgb_est.get_params()["scale_pos_weight"] == 2.0


def test_yaml_null_max_depth_becomes_none():
    params = T.load_params()
    assert None in T.build_grids(params, 1.0, 42)["RandomForest"][1]["max_depth"]


@pytest.fixture
def recorded_train(tmp_path, monkeypatch):
    """Run train() with run_grid and MLflow stubbed; record what reaches run_grid."""
    calls = []
    fake_metrics = {"LogisticRegression": 0.70, "RandomForest": 0.90, "XGBoost": 0.80}

    def fake_run_grid(name, estimator, grid, *data, cv_folds, scoring, random_state):
        calls.append(dict(name=name, grid=grid, cv_folds=cv_folds,
                          scoring=scoring, random_state=random_state))
        m = {k: 0.5 for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
        m[scoring] = fake_metrics[name]
        return {"model": name}, m, f"run-{name}"

    X = pd.DataFrame(np.zeros((4, 2)), columns=["a", "b"])
    y = pd.Series([0, 1, 0, 1])
    monkeypatch.setattr(T, "run_grid", fake_run_grid)
    monkeypatch.setattr(T, "load_data", lambda: (X, X, y, y))
    monkeypatch.setattr(T.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(T.mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(T.joblib, "dump", lambda *a, **k: None)
    monkeypatch.setattr(T, "MODEL_DIR", tmp_path)
    return calls, tmp_path


def test_train_passes_params_to_every_grid(recorded_train):
    calls, _ = recorded_train
    params = dict(TINY_PARAMS, evaluation={"primary_metric": "recall", "cv_folds": 3})
    T.train(params)
    assert [c["name"] for c in calls] == ["LogisticRegression", "RandomForest", "XGBoost"]
    for c in calls:
        assert c["cv_folds"] == 3
        assert c["scoring"] == "recall"
        assert c["random_state"] == TINY_PARAMS["data"]["random_state"]


def test_champion_selected_by_primary_metric(recorded_train):
    _, model_dir = recorded_train
    T.train(TINY_PARAMS)
    info = json.loads((model_dir / "champion_info.json").read_text())
    assert info["champion"] == "RandomForest"
    assert info["feature_names"] == ["a", "b"]
    # paths are stored relative (file names), not machine-specific absolute paths
    assert info["all_models"]["XGBoost"]["path"] == "xgboost_model.joblib"


def test_train_without_argument_reads_params_file(recorded_train, monkeypatch):
    calls, _ = recorded_train
    monkeypatch.setattr(T, "load_params", lambda: TINY_PARAMS)
    T.train()
    assert calls[0]["grid"] == {"C": [1.0]}
