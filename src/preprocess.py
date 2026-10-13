"""
DVC Stage 1: preprocess
Input:  data/raw/heart_disease_uci.csv, params.yaml
Output: data/processed/  (splits, scaler, feature names)

Run from project root: python src/preprocess.py
"""
import json
import joblib
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:                                     # imported as a package (tests, API)
    from src import features as F
except ImportError:                      # run as a script: python src/preprocess.py
    import features as F

# ── Absolute paths (works from any cwd) ──────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW         = ROOT / "data" / "raw" / "heart_disease_uci.csv"
PROC        = ROOT / "data" / "processed"
PARAMS_FILE = ROOT / "params.yaml"

# Re-exported for backwards compatibility
CP_VALS, RESTECG_VALS, SLOPE_VALS = F.CP_VALS, F.RESTECG_VALS, F.SLOPE_VALS
THAL_VALS, DATASET_VALS, SEX_VALS = F.THAL_VALS, F.DATASET_VALS, F.SEX_VALS
BOOL_MAP = F.BOOL_MAP


def load_params(path: Path = None) -> dict:
    with open(path or PARAMS_FILE) as fh:
        return yaml.safe_load(fh)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path.name}")
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: numeric → median, bool → 0, categorical → mode."""
    df = df.copy()
    before = df.isnull().sum().sum()
    for c in F.NUM_COLS:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            df[c] = col.fillna(col.median())
    for c in F.BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].map(F.BOOL_MAP).fillna(0).astype(int)
    for c in F.CAT_COLS:
        if c in df.columns and df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    after = df.isnull().sum().sum()
    print(f"  Imputed {int(before - after)} missing values")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns against the fixed category lists,
    so the output columns do not depend on which values appear in the data."""
    df = df.copy()
    existing = [c for c in F.CAT_COLS if c in df.columns]
    for c in existing:
        df[c] = pd.Categorical(df[c], categories=sorted(F.CATEGORIES[c]))
    return pd.get_dummies(df, columns=existing, drop_first=False)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated interaction features (see src/features.py)."""
    df = F.engineer(df)
    print(f"  Engineered features → {df.shape[1]} total columns")
    return df


def preprocess(params: dict = None):
    params = params or load_params()
    data_p = params.get("data", {})
    target_col   = data_p.get("target_column", "num")
    test_size    = data_p.get("test_size", 0.2)
    random_state = data_p.get("random_state", 42)

    PROC.mkdir(parents=True, exist_ok=True)
    df = load(RAW)

    # Drop id — not a feature
    df = df.drop(columns=["id"], errors="ignore")

    # Binarize target: num > 0 → disease (1), num == 0 → no disease (0)
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
        df, y, test_size=test_size, random_state=random_state, stratify=y
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
    y_train.to_csv(  PROC / "y_train.csv", index=False)
    y_test.to_csv(   PROC / "y_test.csv",  index=False)
    with open(PROC / "feature_names.json", "w") as fh:
        json.dump(list(X_train.columns), fh)

    print(f"\nSaved to {PROC}/")
    print(f"  Train : {X_train_s.shape}")
    print(f"  Test  : {X_test_s.shape}")
    print(f"  Features: {len(X_train.columns)}")


if __name__ == "__main__":
    preprocess()
