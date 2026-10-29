"""
Week 8 (Ali) — property-based tests (Hypothesis) for the feature logic and
for training-vs-serving consistency.

Instead of hand-picked examples, Hypothesis generates hundreds of inputs and
checks properties that must hold for all of them:

  * age_group is always in {0,1,2,3} and never decreases as age increases
  * hr_age_ratio is finite and non-negative for non-negative inputs
  * risk_label never goes down as probability goes up
  * safe_float always returns a finite number
  * preprocess_row always yields a finite vector of the right width
  * for any complete valid patient, the API encodes it exactly like training
"""
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from api.main import risk_label, safe_float
from src import features as F
from src import preprocess as P

RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}
FAST = settings(max_examples=300, deadline=None)
SLOW = settings(max_examples=60, deadline=None,
                suppress_health_check=[HealthCheck.function_scoped_fixture])

ages = st.floats(min_value=0, max_value=150, allow_nan=False)


# ── pure feature functions ──────────────────────────────────────────────────
@FAST
@given(ages)
def test_age_group_in_range(age):
    assert F.age_group(age) in {0, 1, 2, 3}


@FAST
@given(ages, ages)
def test_age_group_monotonic(a, b):
    lo, hi = sorted((a, b))
    assert F.age_group(lo) <= F.age_group(hi)


@FAST
@given(st.floats(0, 250, allow_nan=False), st.floats(1, 150, allow_nan=False))
def test_hr_age_ratio_finite_non_negative(thalch, age):
    # Ages below 1 are outside NUMERIC_RANGES (rejected by validation); with a
    # sub-normal age such as 2e-311 the division overflows to inf.
    r = F.hr_age_ratio(thalch, age)
    assert math.isfinite(r) and r >= 0


@FAST
@given(st.floats(0, 1), st.floats(0, 1))
def test_risk_label_monotonic(p, q):
    lo, hi = sorted((p, q))
    assert RISK_ORDER[risk_label(lo)] <= RISK_ORDER[risk_label(hi)]


@FAST
@given(st.one_of(st.none(), st.floats(), st.integers(), st.text(max_size=8),
                 st.sampled_from(["nan", "NaN", "inf", "-inf", "1e400"])))
def test_safe_float_always_finite(val):
    """Found a defect: the strings "nan"/"inf" used to come back as NaN/inf."""
    assert math.isfinite(safe_float(val, default=1.0))


@FAST
@given(st.sampled_from(F.CAT_COLS).flatmap(
    lambda c: st.tuples(st.just(c), st.sampled_from(F.CATEGORIES[c]))),
    st.sampled_from([str.lower, str.upper, str.title, lambda s: f"  {s} "]))
def test_canonical_category_ignores_case_and_spaces(col_val, transform):
    col, val = col_val
    assert F.canonical_category(col, transform(val)) == val


# ── strategies for whole patients ───────────────────────────────────────────
def _num(name, integer=False):
    lo, hi = F.NUMERIC_RANGES[name]
    if integer:
        return st.integers(int(lo), int(hi))
    return st.floats(lo, hi, allow_nan=False).map(lambda x: round(x, 1))


patients = st.fixed_dictionaries({
    "age": _num("age", integer=True), "trestbps": _num("trestbps"),
    "chol": _num("chol"), "thalch": _num("thalch"), "oldpeak": _num("oldpeak"),
    "ca": _num("ca", integer=True),
    "fbs": st.booleans(), "exang": st.booleans(),
    **{c: st.sampled_from(F.CATEGORIES[c]) for c in F.CAT_COLS},
})

messy_values = st.one_of(st.none(), st.just(float("nan")), st.just(""),
                         st.text(max_size=6), st.floats(), st.integers(-10**6, 10**6))
messy_rows = st.dictionaries(
    st.sampled_from(F.NUM_COLS + F.BOOL_COLS + F.CAT_COLS), messy_values)


@pytest.fixture(scope="module")
def loaded_api(pipeline_artifacts):
    import api.main as api
    mp = pytest.MonkeyPatch()
    mp.setattr(api, "PROC_DIR", pipeline_artifacts.proc_dir)
    mp.setattr(api, "MODEL_DIR", pipeline_artifacts.model_dir)
    api.load_artifacts()
    yield api
    mp.undo()


@pytest.mark.slow
@SLOW
@given(row=messy_rows)
def test_preprocess_row_total_and_finite(loaded_api, row):
    """Any row, however malformed, encodes to a finite vector of full width."""
    X = loaded_api.preprocess_row(row)
    assert X.shape == (1, len(loaded_api.FEATURES))
    assert np.isfinite(X).all()


@pytest.mark.slow
@SLOW
@given(patient=patients)
def test_training_and_serving_encode_identically(loaded_api, patient):
    """For any complete, valid patient the API's feature vector equals the one
    the training pipeline (impute → encode → engineer → scale) produces."""
    raw = pd.DataFrame([patient])
    trained = P.engineer(P.encode(P.impute(raw)))[loaded_api.FEATURES]
    expected = loaded_api.SCALER.transform(trained)[0]
    np.testing.assert_allclose(loaded_api.preprocess_row(patient)[0], expected, atol=1e-9)


@pytest.mark.slow
@SLOW
@given(patient=patients, name=st.sampled_from(["LogisticRegression", "RandomForest",
                                               "XGBoost"]))
def test_probability_is_a_probability(loaded_api, patient, name):
    pred, prob = loaded_api.score(loaded_api.MODELS[name], loaded_api.preprocess_row(patient))
    assert pred in (0, 1) and 0.0 <= prob <= 1.0
    assert pred == int(prob >= 0.5)
