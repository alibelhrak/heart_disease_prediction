"""
DVC Stage 2: train
Input:  data/processed/ splits
Output: models/  (all 3 models + champion_info.json)

Run from project root: python src/train.py
"""
import json, joblib, mlflow, mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)
import xgboost as xgb

# ── Absolute paths ────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC      = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MLRUNS    = ROOT / "mlruns"

EXPERIMENT = "heart-disease-mlops"


def load_data():
    X_train = pd.read_csv(PROC / "X_train.csv")
    X_test  = pd.read_csv(PROC / "X_test.csv")
    y_train = pd.read_csv(PROC / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PROC / "y_test.csv").squeeze()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def run_grid(name, estimator, param_grid, X_train, y_train, X_test, y_test):
    """GridSearchCV + MLflow logging. Returns (best_model, metrics, run_id)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n{'='*55}")
    print(f"  Training: {name}")

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds", 5)

        grid = GridSearchCV(
            estimator, param_grid,
            scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
        mlflow.log_metric("cv_best_f1", round(float(grid.best_score_), 4))

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, artifact_path="model")

        print(f"  Best params : {grid.best_params_}")
        print(f"  Test metrics: {metrics}")
        print(classification_report(y_test, y_pred,
              target_names=["No Disease", "Disease"], zero_division=0))

        return best, metrics, run.info.run_id


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # MLflow — store runs next to project
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    X_train, X_test, y_train, y_test = load_data()
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    results = {}

    # ── 1. Logistic Regression ────────────────────────────────────────────
    lr, lr_m, lr_rid = run_grid(
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100]},
        X_train, y_train, X_test, y_test,
    )
    results["LogisticRegression"] = {"model": lr, "metrics": lr_m, "run_id": lr_rid}

    # ── 2. Random Forest ──────────────────────────────────────────────────
    rf, rf_m, rf_rid = run_grid(
        "RandomForest",
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [4, 6, None], "min_samples_split": [2, 5]},
        X_train, y_train, X_test, y_test,
    )
    results["RandomForest"] = {"model": rf, "metrics": rf_m, "run_id": rf_rid}

    # ── 3. XGBoost ────────────────────────────────────────────────────────
    xg, xg_m, xg_rid = run_grid(
        "XGBoost",
        xgb.XGBClassifier(
            random_state=42, scale_pos_weight=spw,
            eval_metric="logloss", use_label_encoder=False,
        ),
        {"n_estimators": [100, 200], "max_depth": [3, 5],
         "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
        X_train, y_train, X_test, y_test,
    )
    results["XGBoost"] = {"model": xg, "metrics": xg_m, "run_id": xg_rid}

    # ── Select champion by F1 ────────────────────────────────────────────
    champion_name = max(results, key=lambda k: results[k]["metrics"]["f1"])
    champ = results[champion_name]

    print(f"\n{'='*55}")
    print(f"  CHAMPION : {champion_name}")
    print(f"  Metrics  : {champ['metrics']}")
    print(f"{'='*55}")

    # ── Save all individual models ────────────────────────────────────────
    all_models_info = {}
    for name, res in results.items():
        path = MODEL_DIR / f"{name.lower()}_model.joblib"
        joblib.dump(res["model"], path)
        all_models_info[name] = {
            "metrics": res["metrics"],
            "run_id":  res["run_id"],
            "path":    str(path),
        }
        print(f"  Saved {name} → {path.name}")

    # ── Save champion ─────────────────────────────────────────────────────
    joblib.dump(champ["model"], MODEL_DIR / "champion_model.joblib")

    champion_info = {
        "champion":       champion_name,
        "run_id":         champ["run_id"],
        "metrics":        champ["metrics"],
        "all_models":     all_models_info,
        "feature_names":  list(X_train.columns),
    }
    json.dump(champion_info, open(MODEL_DIR / "champion_info.json", "w"), indent=2)
    print(f"\nChampion saved → models/champion_model.joblib")
    print(f"Metadata saved → models/champion_info.json")

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 57)
    for name, res in results.items():
        m    = res["metrics"]
        mark = " ★" if name == champion_name else ""
        print(f"{name:<25} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f}{mark}")


if __name__ == "__main__":
    train()
