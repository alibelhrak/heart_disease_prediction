"""Week 7 (Ali) — tests for the shared feature module src/features.py."""
import numpy as np
import pandas as pd
import pytest

from src import features as F


@pytest.mark.parametrize("age, group", [(0, 0), (29, 0), (45, 0), (45.5, 1), (55, 1),
                                        (55.01, 2), (65, 2), (66, 3), (120, 3), (150, 3)])
def test_age_group_scalar(age, group):
    result = F.age_group(age)
    assert result == group and isinstance(result, int)


def test_age_group_matches_original_pd_cut_definition():
    ages = pd.Series(np.arange(1, 121), dtype=float)
    reference = pd.cut(ages, bins=[0, 45, 55, 65, 120], labels=[0, 1, 2, 3]).astype(int)
    pd.testing.assert_series_equal(F.age_group(ages), reference, check_names=False)


def test_age_group_series_keeps_index():
    s = pd.Series([40.0, 70.0], index=[10, 20])
    out = F.age_group(s)
    assert list(out.index) == [10, 20] and out.tolist() == [0, 3]


def test_age_group_ndarray():
    out = F.age_group(np.array([40, 50, 60, 70]))
    assert isinstance(out, np.ndarray) and out.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("thalch, age, expected", [(150, 50, 3.0), (150, 0, 0.0),
                                                   (150, -5, 0.0), (0, 40, 0.0)])
def test_hr_age_ratio_scalar(thalch, age, expected):
    assert F.hr_age_ratio(thalch, age) == pytest.approx(expected)


def test_hr_age_ratio_vectorised_no_warnings():
    with np.errstate(all="raise"):           # division by zero must be avoided
        out = F.hr_age_ratio(np.array([150, 150]), np.array([50, 0]))
    assert out.tolist() == [3.0, 0.0]


def test_engineer_does_not_mutate_input():
    df = pd.DataFrame({"age": [50.0], "thalch": [150.0], "oldpeak": [1.0]})
    before = df.copy()
    F.engineer(df)
    pd.testing.assert_frame_equal(df, before)


def test_category_lists_are_consistent():
    assert set(F.CATEGORIES) == set(F.CAT_COLS)
    for values in F.CATEGORIES.values():
        assert len(values) == len(set(values))


def test_api_and_training_share_the_same_definitions():
    import api.main as api
    from src import preprocess as P
    for name in ("CP_VALS", "RESTECG_VALS", "SLOPE_VALS", "THAL_VALS",
                 "DATASET_VALS", "SEX_VALS", "BOOL_MAP"):
        assert getattr(api, name) is getattr(F, name)
        assert getattr(P, name) is getattr(F, name)
