"""Week 5 (Ali) — branch tests for src/preprocess.py: impute, encode, engineer."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.preprocess import encode, engineer, impute


# ── impute ──────────────────────────────────────────────────────────────────
class TestImpute:
    def test_numeric_nan_filled_with_median(self):
        df = pd.DataFrame({"chol": [100.0, np.nan, 300.0]})
        assert impute(df)["chol"].tolist() == [100.0, 200.0, 300.0]

    def test_numeric_non_numeric_string_coerced_then_filled(self):
        df = pd.DataFrame({"age": ["40", "bad", "60"]})
        assert impute(df)["age"].tolist() == [40.0, 50.0, 60.0]

    def test_numeric_column_absent_is_skipped(self):
        df = pd.DataFrame({"other": [1, 2]})
        out = impute(df)
        assert list(out.columns) == ["other"]

    @pytest.mark.parametrize("raw, expected", [
        (True, 1), (False, 0), ("True", 1), ("False", 0),
        ("true", 1), ("false", 0), (1, 1), (0, 0),
    ])
    def test_bool_mapping(self, raw, expected):
        df = pd.DataFrame({"fbs": [raw]}, dtype=object)
        assert impute(df)["fbs"].iloc[0] == expected

    def test_bool_missing_or_unknown_becomes_zero(self):
        df = pd.DataFrame({"exang": [np.nan, "maybe", True]}, dtype=object)
        out = impute(df)["exang"]
        assert out.tolist() == [0, 0, 1]
        assert out.dtype.kind == "i"

    def test_categorical_nan_filled_with_mode(self):
        df = pd.DataFrame({"thal": ["normal", "normal", "fixed defect", np.nan]})
        assert impute(df)["thal"].tolist()[-1] == "normal"

    def test_categorical_without_nan_unchanged(self):
        df = pd.DataFrame({"cp": ["asymptomatic", "non-anginal"]})
        assert impute(df)["cp"].tolist() == ["asymptomatic", "non-anginal"]

    def test_no_missing_values_remain(self, raw_df):
        assert impute(raw_df).drop(columns=["num"]).isnull().sum().sum() == 0

    def test_input_frame_not_mutated(self, raw_df):
        before = raw_df.copy()
        impute(raw_df)
        pd.testing.assert_frame_equal(raw_df, before)


# ── encode ──────────────────────────────────────────────────────────────────
class TestEncode:
    def test_all_categories_produce_columns(self, raw_df):
        out = encode(impute(raw_df))
        for col, values in F.CATEGORIES.items():
            assert col not in out.columns
            for v in values:
                assert f"{col}_{v}" in out.columns

    def test_category_absent_from_data_still_gets_a_column(self):
        # Only 'Male' present; 'sex_Female' must still exist (fixed categories)
        out = encode(pd.DataFrame({"sex": ["Male", "Male"]}))
        assert out["sex_Female"].sum() == 0
        assert out["sex_Male"].sum() == 2

    def test_columns_independent_of_data_subset(self, raw_df):
        full = encode(impute(raw_df)).columns
        subset = encode(impute(raw_df.head(3))).columns
        assert list(full) == list(subset)

    def test_unknown_category_encodes_as_all_zeros(self):
        out = encode(pd.DataFrame({"cp": ["something new"]}))
        assert out.filter(like="cp_").sum(axis=1).iloc[0] == 0

    def test_absent_categorical_column_is_skipped(self):
        out = encode(pd.DataFrame({"age": [50]}))
        assert list(out.columns) == ["age"]


# ── engineer ────────────────────────────────────────────────────────────────
def _encoded(**cols):
    base = {"age": [50.0], "thalch": [150.0], "oldpeak": [1.0]}
    base.update(cols)
    return pd.DataFrame(base)


class TestEngineer:
    @pytest.mark.parametrize("age, group", [
        (30, 0), (45, 0), (46, 1), (55, 1), (56, 2), (65, 2), (66, 3), (90, 3),
    ])
    def test_age_group_bin_edges(self, age, group):
        assert engineer(_encoded(age=[float(age)]))["age_group"].iloc[0] == group

    def test_hr_age_ratio(self):
        assert engineer(_encoded())["hr_age_ratio"].iloc[0] == pytest.approx(3.0)

    def test_hr_age_ratio_zero_age_branch(self):
        assert engineer(_encoded(age=[0.0]))["hr_age_ratio"].iloc[0] == 0

    def test_st_slope_interaction_with_flat_slope(self):
        out = engineer(_encoded(oldpeak=[2.5], slope_flat=[True]))
        assert out["st_slope_interaction"].iloc[0] == pytest.approx(2.5)

    def test_st_slope_interaction_when_slope_flat_missing(self):
        assert engineer(_encoded(oldpeak=[2.5]))["st_slope_interaction"].iloc[0] == 0

    def test_cp_severe_present_and_missing(self):
        assert engineer(_encoded(cp_asymptomatic=[True]))["cp_severe"].iloc[0] == 1
        assert engineer(_encoded())["cp_severe"].iloc[0] == 0

    def test_adds_exactly_the_engineered_columns(self):
        before = _encoded()
        after = engineer(before)
        assert set(after.columns) - set(before.columns) == set(F.ENGINEERED)


# ── params + script entry point ─────────────────────────────────────────────
def test_load_params_default_and_custom(tmp_path):
    from src.preprocess import load_params
    assert load_params()["data"]["target_column"] == "num"
    f = tmp_path / "p.yaml"
    f.write_text("data:\n  test_size: 0.3\n")
    assert load_params(f) == {"data": {"test_size": 0.3}}


def test_runs_as_script_without_src_package():
    """DVC runs `python src/preprocess.py`; then `src` is not importable and
    the module must fall back to `import features`."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    src_dir = Path(__file__).resolve().parents[2] / "src"
    out = subprocess.run([sys.executable, "-c",
                          "import preprocess; print(preprocess.F.__name__)"],
                         cwd=src_dir, capture_output=True, text=True, check=True,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert out.stdout.strip() == "features"
