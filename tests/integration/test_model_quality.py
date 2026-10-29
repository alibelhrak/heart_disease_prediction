"""
Week 9 (Ali) — model-quality and determinism tests.

Quality is checked on the project's own 920-row dataset with the real
params.yaml grids (the `full_pipeline` fixture, ~20 s). Determinism is checked
by running the whole pipeline twice with the same seed.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import check_threshold as gate
from tests.conftest import TINY_PARAMS, make_raw_df, run_pipeline

pytestmark = pytest.mark.slow


def _load(run):
    info = json.loads((run.model_dir / "champion_info.json").read_text())
    X = pd.read_csv(run.proc_dir / "X_test.csv")
    y = pd.read_csv(run.proc_dir / "y_test.csv").squeeze()
    return info, X, y


# ── quality on the project dataset ──────────────────────────────────────────
def test_every_model_beats_the_majority_class_baseline(full_pipeline):
    info, _, y = _load(full_pipeline)
    majority_acc = max(y.mean(), 1 - y.mean())
    for name, entry in info["all_models"].items():
        assert entry["metrics"]["accuracy"] > majority_acc, name


def test_every_model_ranks_better_than_chance(full_pipeline):
    info, _, _ = _load(full_pipeline)
    for name, entry in info["all_models"].items():
        assert entry["metrics"]["roc_auc"] >= 0.80, name


def test_champion_passes_the_ci_gate(full_pipeline):
    info, _, _ = _load(full_pipeline)
    assert gate.passes(info["metrics"]["f1"], 0.70)


def test_champion_recall_is_clinically_useful(full_pipeline):
    """Missing a sick patient is the costly error: recall must stay high."""
    info, _, _ = _load(full_pipeline)
    assert info["metrics"]["recall"] >= 0.80


def test_models_are_better_than_a_shuffled_label_model(full_pipeline):
    """If labels carried no signal, F1 would not exceed the baseline; the
    champion's predictions must agree with the truth far more than chance."""
    info, X, y = _load(full_pipeline)
    model = joblib.load(full_pipeline.model_dir / "champion_model.joblib")
    acc = (model.predict(X) == y.values).mean()
    shuffled = np.random.default_rng(0).permutation(y.values)
    assert acc - (model.predict(X) == shuffled).mean() > 0.15


# ── determinism ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory):
    raw = make_raw_df(n=160, seed=4)
    a = run_pipeline(tmp_path_factory.mktemp("a"), raw, TINY_PARAMS)
    b = run_pipeline(tmp_path_factory.mktemp("b"), raw, TINY_PARAMS)
    return a, b


def test_same_seed_gives_identical_splits(twin_runs):
    a, b = twin_runs
    for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
                 "feature_names.json"):
        assert (a.proc_dir / name).read_bytes() == (b.proc_dir / name).read_bytes(), name


def test_same_seed_gives_identical_models(twin_runs):
    (info_a, X, _), (info_b, _, _) = _load(twin_runs[0]), _load(twin_runs[1])
    assert info_a["champion"] == info_b["champion"]
    for name in info_a["all_models"]:
        assert info_a["all_models"][name]["metrics"] == info_b["all_models"][name]["metrics"]
        pa = joblib.load(twin_runs[0].model_dir / f"{name.lower()}_model.joblib")
        pb = joblib.load(twin_runs[1].model_dir / f"{name.lower()}_model.joblib")
        np.testing.assert_array_equal(pa.predict_proba(X), pb.predict_proba(X))


def test_different_seed_changes_the_split(tmp_path):
    raw = make_raw_df(n=160, seed=4)
    params = {**TINY_PARAMS, "data": {**TINY_PARAMS["data"], "random_state": 99}}
    other = run_pipeline(tmp_path / "c", raw, params)
    base = run_pipeline(tmp_path / "d", raw, TINY_PARAMS)
    assert (other.proc_dir / "y_test.csv").read_bytes() != \
        (base.proc_dir / "y_test.csv").read_bytes()
