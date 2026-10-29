"""
Week 8 (Mohaned) — equivalence partitioning for categorical and boolean inputs
of api.main.preprocess_row.

  Class                          Example              Expected encoding
  -----------------------------  -------------------  -----------------------------
  V1 valid, canonical spelling   "asymptomatic"       exactly that one-hot bit = 1
  V2 valid, other case/spacing   " MALE ", "flat "    same as V1 (canonicalised)
  V3 missing (None/NaN/blank)    None, nan, "  "      the column's default category
  I1 unknown value               "Other", "x"         all bits 0 for that column
  Booleans: true-like {True,"true","1",1}, false-like {False,"false","0",0},
            missing/unknown {None,"maybe"} → 0
"""
import numpy as np
import pytest

from src import features as F

DEFAULTS = {"sex": "Male", "dataset": "Cleveland", "cp": "asymptomatic",
            "restecg": "normal", "slope": "flat", "thal": "normal"}
VALID = [(c, v) for c in F.CAT_COLS for v in F.CATEGORIES[c]]


@pytest.fixture(scope="module")
def api(pipeline_artifacts):
    import api.main as api
    mp = pytest.MonkeyPatch()
    mp.setattr(api, "PROC_DIR", pipeline_artifacts.proc_dir)
    mp.setattr(api, "MODEL_DIR", pipeline_artifacts.model_dir)
    api.load_artifacts()
    yield api
    mp.undo()


def bits(api, row, col):
    """Unscaled one-hot bits for `col`, as a {category: 0/1} dict."""
    X = api.SCALER.inverse_transform(api.preprocess_row({"age": 50, **row}))[0]
    return {v: int(round(X[api.FEATURES.index(f"{col}_{v}")])) for v in F.CATEGORIES[col]}


pytestmark = pytest.mark.slow


@pytest.mark.parametrize("col, val", VALID, ids=[f"{c}={v}" for c, v in VALID])
def test_v1_each_valid_category_sets_exactly_its_bit(api, col, val):
    b = bits(api, {col: val}, col)
    assert b[val] == 1 and sum(b.values()) == 1


@pytest.mark.parametrize("col, raw, canon", [
    ("sex", "male", "Male"), ("sex", " FEMALE ", "Female"),
    ("cp", "Asymptomatic", "asymptomatic"), ("slope", "flat ", "flat"),
    ("dataset", "va long beach", "VA Long Beach"), ("thal", "Fixed Defect", "fixed defect"),
])
def test_v2_case_and_whitespace_variants(api, col, raw, canon):
    """Defect found: lowercase "male" used to encode as neither Male nor Female."""
    assert bits(api, {col: raw}, col) == bits(api, {col: canon}, col)


@pytest.mark.parametrize("missing", [None, float("nan"), "", "   "])
@pytest.mark.parametrize("col", F.CAT_COLS)
def test_v3_missing_uses_default(api, col, missing):
    b = bits(api, {col: missing}, col)
    assert b[DEFAULTS[col]] == 1 and sum(b.values()) == 1


@pytest.mark.parametrize("col", F.CAT_COLS)
@pytest.mark.parametrize("unknown", ["Other", "x", "0"])
def test_i1_unknown_value_sets_no_bit(api, col, unknown):
    # Unknown categories are rejected up front by the JSON schema and by the
    # CSV validation layer (Week 9); at this level they encode as all zeros.
    assert sum(bits(api, {col: unknown}, col).values()) == 0


def _bool_bit(api, col, val):
    X = api.SCALER.inverse_transform(api.preprocess_row({"age": 50, col: val}))[0]
    return round(X[api.FEATURES.index(col)])


@pytest.mark.parametrize("col", F.BOOL_COLS)
@pytest.mark.parametrize("val, bit", [(True, 1), ("true", 1), ("True", 1), ("1", 1), (1, 1),
                                      (False, 0), ("false", 0), ("0", 0), (0, 0),
                                      (None, 0), ("maybe", 0)])
def test_boolean_partitions(api, col, val, bit):
    assert _bool_bit(api, col, val) == bit


@pytest.mark.parametrize("val", [None, float("nan"), "Other", ""])
def test_canonical_category_returns_none_outside_valid_classes(val):
    assert F.canonical_category("sex", val) is None
