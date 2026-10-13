"""Week 5 (Mohaned) — branch tests for api/main.py helpers."""
import numpy as np
import pytest

from api.main import cat_or_default, interpret, ohe, risk_label, safe_float


# ── safe_float ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("val, expected", [
    (None, 7.0),            # None branch
    (float("nan"), 7.0),    # NaN branch
    (np.nan, 7.0),
    ("abc", 7.0),           # ValueError branch
    ([1, 2], 7.0),          # TypeError branch
    ("12.5", 12.5),         # numeric string
    (3, 3.0),               # int
    (0.0, 0.0),             # falsy but valid
])
def test_safe_float(val, expected):
    assert safe_float(val, default=7.0) == expected


def test_safe_float_default_default():
    assert safe_float(None) == 0.0


# ── ohe ─────────────────────────────────────────────────────────────────────
VALS = ["normal", "flat"]


def test_ohe_match():
    assert ohe("flat", VALS, "slope") == {"slope_normal": 0, "slope_flat": 1}


def test_ohe_strips_whitespace():
    assert ohe("  flat ", VALS, "slope")["slope_flat"] == 1


def test_ohe_no_match_all_zero():
    assert sum(ohe("unknown", VALS, "slope").values()) == 0


def test_ohe_nan_all_zero():
    assert sum(ohe(float("nan"), VALS, "slope").values()) == 0


# ── cat_or_default ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("row", [{}, {"cp": None}, {"cp": float("nan")}, {"cp": "  "}])
def test_cat_or_default_falls_back(row):
    assert cat_or_default(row, "cp", "asymptomatic") == "asymptomatic"


def test_cat_or_default_keeps_value():
    assert cat_or_default({"cp": "non-anginal"}, "cp", "asymptomatic") == "non-anginal"


# ── risk_label / interpret thresholds ───────────────────────────────────────
@pytest.mark.parametrize("p, label", [(0.0, "Low"), (0.399, "Low"), (0.40, "Medium"),
                                      (0.699, "Medium"), (0.70, "High"), (1.0, "High")])
def test_risk_label_thresholds(p, label):
    assert risk_label(p) == label
    assert interpret(p).startswith({"Low": "Low", "Medium": "Moderate",
                                    "High": "High"}[label])


# ── preprocess_row (needs scaler + feature list from the tmp pipeline) ─────
@pytest.fixture
def loaded_api(api_module):
    api_module.load_artifacts()
    return api_module


def test_preprocess_row_shape(loaded_api):
    X = loaded_api.preprocess_row({"age": 50, "sex": "Male", "cp": "asymptomatic"})
    assert X.shape == (1, len(loaded_api.FEATURES))
    assert np.isfinite(X).all()


def test_preprocess_row_empty_dict_uses_defaults(loaded_api):
    defaults = {"age": 54, "thalch": 150, "oldpeak": 0.0, "trestbps": 130,
                "chol": 240, "ca": 0, "sex": "Male", "dataset": "Cleveland",
                "cp": "asymptomatic", "restecg": "normal", "slope": "flat",
                "thal": "normal"}
    np.testing.assert_allclose(loaded_api.preprocess_row({}),
                               loaded_api.preprocess_row(defaults))


def test_preprocess_row_nan_categorical_uses_default(loaded_api):
    with_nan = loaded_api.preprocess_row({"age": 60, "thal": float("nan")})
    explicit = loaded_api.preprocess_row({"age": 60, "thal": "normal"})
    np.testing.assert_allclose(with_nan, explicit)


def test_preprocess_row_garbage_numbers_do_not_crash(loaded_api):
    X = loaded_api.preprocess_row({"age": "old", "chol": None, "fbs": "maybe"})
    assert np.isfinite(X).all()


def test_preprocess_row_zero_age(loaded_api):
    X = loaded_api.preprocess_row({"age": 0})
    idx = loaded_api.FEATURES.index("hr_age_ratio")
    unscaled = loaded_api.SCALER.inverse_transform(X)[0, idx]
    assert unscaled == pytest.approx(0.0, abs=1e-9)
