"""
DVC Stage 1: preprocess
Input:  data/raw/heart_disease_uci.csv
Output: data/processed/  (splits, scaler, feature names)

Run from project root: python src/preprocess.py
"""
import json, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Absolute paths (works from any cwd) ──────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
RAW   = ROOT / "data" / "raw" / "heart_disease_uci.csv"
PROC  = ROOT / "data" / "processed"

# ── Encoding maps ─────────────────────────────────────────────────────────────
CP_VALS      = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
RESTECG_VALS = ["normal", "st-t abnormality", "lv hypertrophy"]
SLOPE_VALS   = ["upsloping", "flat", "downsloping"]
THAL_VALS    = ["normal", "fixed defect", "reversable defect"]
DATASET_VALS = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]
SEX_VALS     = ["Male", "Female"]
BOOL_MAP     = {True: 1, False: 0, "True": 1, "False": 0,
                "true": 1, "false": 0, 1: 1, 0: 0}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path.name}")
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: numeric → median, bool → 0, categorical → mode."""
    num_cols  = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    bool_cols = ["fbs", "exang"]
    cat_cols  = ["sex", "dataset", "cp", "restecg", "slope", "thal"]

    before = df.isnull().sum().sum()
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map(BOOL_MAP).fillna(0).astype(int)
    for c in cat_cols:
        if c in df.columns and df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    after = df.isnull().sum().sum()
    print(f"  Imputed {int(before - after)} missing values")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns."""
    cat_cols = ["sex", "dataset", "cp", "restecg", "slope", "thal"]
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=False)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated interaction features."""
    df["age_group"]            = pd.cut(df["age"], bins=[0,45,55,65,120],
                                         labels=[0,1,2,3]).astype(int)
    df["hr_age_ratio"]         = df["thalch"] / df["age"].replace(0, np.nan)
    df["hr_age_ratio"]         = df["hr_age_ratio"].fillna(0)
    df["st_slope_interaction"] = df["oldpeak"] * df.get("slope_flat",
                                                         pd.Series(0, index=df.index))
    df["cp_severe"]            = df.get("cp_asymptomatic",
                                        pd.Series(0, index=df.index))
    print(f"  Engineered features → {df.shape[1]} total columns")
    return df


def preprocess():
    PROC.mkdir(parents=True, exist_ok=True)

    df = load(RAW)

    # Drop id — not a feature
    df = df.drop(columns=["id"], errors="ignore")

    # Binarize target: num > 0 → disease (1), num == 0 → no disease (0)
    target_col = "num"
    y = (df[target_col] > 0).astype(int)
    df = df.drop(columns=[target_col])

    print(f"\nClass distribution: {dict(y.value_counts().sort_index())}")
    print(f"Disease prevalence: {y.mean():.1%}")

    # Clean → encode → engineer
    df = impute(df)
    df = encode(df)
    df = engineer(df)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale — fit on train only (no data leakage)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns, index=X_test.index)

    # Save everything
    joblib.dump(scaler, PROC / "scaler.joblib")
    X_train_s.to_csv(PROC / "X_train.csv", index=False)
    X_test_s.to_csv( PROC / "X_test.csv",  index=False)
    y_train.to_csv(  PROC / "y_train.csv",  index=False)
    y_test.to_csv(   PROC / "y_test.csv",   index=False)
    json.dump(list(X_train.columns), open(PROC / "feature_names.json", "w"))

    print(f"\nSaved to {PROC}/")
    print(f"  Train : {X_train_s.shape}")
    print(f"  Test  : {X_test_s.shape}")
    print(f"  Features: {len(X_train.columns)}")


if __name__ == "__main__":
    preprocess()
