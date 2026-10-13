"""
Single source of truth for feature definitions.

Both the training pipeline (src/preprocess.py) and the inference API
(api/main.py) import from here, so the engineered features are computed
identically at training time and at serving time.

Before this module existed, the API computed age_group as (age - 30) // 10
while training used pd.cut with bins [0, 45, 55, 65, 120]. Patients aged
e.g. 40, 50 or 60 therefore received a different feature value at inference
than the model saw during training (training/serving skew).
"""
import numpy as np
import pandas as pd

# ── Raw column groups ────────────────────────────────────────────────────────
NUM_COLS  = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
BOOL_COLS = ["fbs", "exang"]
CAT_COLS  = ["sex", "dataset", "cp", "restecg", "slope", "thal"]

# ── Allowed categorical values ───────────────────────────────────────────────
CP_VALS      = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
RESTECG_VALS = ["normal", "st-t abnormality", "lv hypertrophy"]
SLOPE_VALS   = ["upsloping", "flat", "downsloping"]
THAL_VALS    = ["normal", "fixed defect", "reversable defect"]
DATASET_VALS = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]
SEX_VALS     = ["Male", "Female"]

CATEGORIES = {
    "sex":     SEX_VALS,
    "dataset": DATASET_VALS,
    "cp":      CP_VALS,
    "restecg": RESTECG_VALS,
    "slope":   SLOPE_VALS,
    "thal":    THAL_VALS,
}

BOOL_MAP = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0,
            1: 1, 0: 0, "1": 1, "0": 0}

# Right-inclusive upper edges: (0,45] -> 0, (45,55] -> 1, (55,65] -> 2, >65 -> 3
AGE_EDGES = [45, 55, 65]

ENGINEERED = ["age_group", "hr_age_ratio", "st_slope_interaction", "cp_severe"]


def age_group(age):
    """Ordinal age bucket. Works on scalars, numpy arrays and pandas Series."""
    groups = np.searchsorted(AGE_EDGES, np.asarray(age, dtype=float), side="left")
    if isinstance(age, pd.Series):
        return pd.Series(groups.astype(int), index=age.index)
    if np.ndim(groups) == 0:
        return int(groups)
    return groups.astype(int)


def hr_age_ratio(thalch, age):
    """Max heart rate divided by age; 0 when age is not positive."""
    thalch = np.asarray(thalch, dtype=float)
    age = np.asarray(age, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(age > 0, thalch / np.where(age > 0, age, 1.0), 0.0)
    return float(ratio) if ratio.ndim == 0 else ratio


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features to an already one-hot-encoded frame."""
    df = df.copy()
    zeros = pd.Series(0, index=df.index)
    df["age_group"]            = age_group(df["age"])
    df["hr_age_ratio"]         = hr_age_ratio(df["thalch"], df["age"])
    df["st_slope_interaction"] = df["oldpeak"] * df.get("slope_flat", zeros).astype(float)
    df["cp_severe"]            = df.get("cp_asymptomatic", zeros).astype(int)
    return df
