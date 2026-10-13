"""
Week 6 (Ali) — def-use tests for the scaler and the feature list across
preprocess → train → API.

  Variable        Definition (def)                      Uses
  --------------  ------------------------------------  ----------------------------------
  scaler          preprocess(): StandardScaler.fit,     api.load_artifacts() -> SCALER,
                  joblib.dump -> scaler.joblib          preprocess_row(): SCALER.transform
  feature list    preprocess(): X_train.columns ->      train(): X_train.csv header ->
                  feature_names.json, X_train.csv       champion_info["feature_names"],
                                                        model.n_features_in_;
                                                        api: FEATURES -> reindex()

Each test covers at least one def-use pair end to end.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import preprocess as P

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def artifacts(pipeline_artifacts):
    a = pipeline_artifacts
    return {
        "scaler":    joblib.load(a.proc_dir / "scaler.joblib"),
        "features":  json.loads((a.proc_dir / "feature_names.json").read_text()),
        "X_train":   pd.read_csv(a.proc_dir / "X_train.csv"),
        "X_test":    pd.read_csv(a.proc_dir / "X_test.csv"),
        "champion":  json.loads((a.model_dir / "champion_info.json").read_text()),
        "models":    {p.stem: joblib.load(p) for p in a.model_dir.glob("*.joblib")},
        "raw":       pd.read_csv(a.raw_csv),
    }


# ── feature list: preprocess (def) → train / API (uses) ────────────────────
def test_feature_list_written_matches_training_matrix(artifacts):
    assert artifacts["features"] == list(artifacts["X_train"].columns)
    assert artifacts["features"] == list(artifacts["X_test"].columns)


def test_feature_list_reaches_champion_info(artifacts):
    assert artifacts["champion"]["feature_names"] == artifacts["features"]


def test_scaler_was_fit_on_the_same_feature_list(artifacts):
    assert list(artifacts["scaler"].feature_names_in_) == artifacts["features"]


def test_every_model_expects_the_feature_count(artifacts):
    for name, model in artifacts["models"].items():
        assert model.n_features_in_ == len(artifacts["features"]), name


def test_api_loads_the_same_feature_list_and_scaler(api_module, artifacts):
    api_module.load_artifacts()
    assert api_module.FEATURES == artifacts["features"]
    np.testing.assert_allclose(api_module.SCALER.mean_, artifacts["scaler"].mean_)


# ── scaler: fit on train only (def) → transform in preprocess / API (uses) ─
def test_scaler_fit_on_training_split_only(artifacts):
    X_train = artifacts["X_train"]
    # Standardised with its own statistics → mean 0, std 1 per column
    np.testing.assert_allclose(X_train.mean(), 0, atol=1e-8)
    non_constant = artifacts["scaler"].scale_ != 1.0
    np.testing.assert_allclose(X_train.std(ddof=0)[non_constant], 1, atol=1e-6)


def test_test_split_transformed_not_refit(artifacts):
    # If the scaler had been refit on the test split, its means would be ~0
    assert not np.allclose(artifacts["X_test"].mean(), 0, atol=1e-8)


# ── training/serving parity: the two uses of the scaler must agree ──────────
def _training_path(raw: pd.DataFrame, scaler, features) -> np.ndarray:
    df = raw.drop(columns=["id", "num"])
    df = P.engineer(P.encode(P.impute(df)))
    return scaler.transform(df[features])


def test_api_preprocess_row_matches_training_pipeline(api_module, artifacts):
    """Same raw patient → same scaled vector, in training and in the API.

    This failed before src/features.py: the API bucketed age_group as
    (age - 30) // 10 while training used bins (0,45], (45,55], (55,65], (65,120].
    """
    api_module.load_artifacts()
    raw = artifacts["raw"].dropna().head(40)   # complete rows: no imputation drift
    expected = _training_path(raw, artifacts["scaler"], artifacts["features"])
    for i, (_, row) in enumerate(raw.iterrows()):
        got = api_module.preprocess_row(row.to_dict())
        np.testing.assert_allclose(got[0], expected[i], atol=1e-9,
                                   err_msg=f"row id={row['id']} age={row['age']}")


@pytest.mark.parametrize("age", [40, 45, 46, 50, 55, 56, 60, 65, 66])
def test_age_group_parity_at_bin_edges(api_module, artifacts, age):
    api_module.load_artifacts()
    row = artifacts["raw"].dropna().iloc[[0]].copy()
    row["age"] = age
    expected = _training_path(row, artifacts["scaler"], artifacts["features"])[0]
    got = api_module.preprocess_row(row.iloc[0].to_dict())[0]
    idx = artifacts["features"].index("age_group")
    assert got[idx] == pytest.approx(expected[idx])
