"""
DVC Stage 2: train
Input:  data/processed/ splits, params.yaml
Output: models/  (all 3 models + champion_info.json)

Run from project root: python src/train.py
"""
import json, os, joblib, mlflow, mlflow.sklearn
import yaml
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
PARAMS_FILE = ROOT / "params.yaml"

EXPERIMENT = "heart-disease-mlops"


def load_params(path: Path = None) -> dict:
    """Read hyperparameters from params.yaml (tracked by DVC)."""
    with open(path or PARAMS_FILE) as fh:
        return yaml.safe_load(fh)


def build_grids(params: dict, spw: float, random_state: int) -> dict:
    """Turn params.yaml sections into (estimator, param_grid) pairs."""
    lr_p, rf_p, xg_p = (params["logistic_regression"], params["random_forest"],
                        params["xgboost"])
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=lr_p.get("max_iter", 1000),
                               random_state=random_state),
            {"C": lr_p["C"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_state),
            {k: rf_p[k] for k in ("n_estimators", "max_depth", "min_samples_split")},
        ),
        "XGBoost": (
            xgb.XGBClassifier(random_state=random_state, scale_pos_weight=spw,
                              eval_metric="logloss"),
            {k: xg_p[k] for k in ("n_estimators", "max_depth",
                                  "learning_rate", "subsample")},
        ),
    }


def load_data():
    X_train = pd.read_csv(PROC / "X_train.csv")
    X_test  = pd.read_csv(PROC / "X_test.csv")
    y_train = pd.read_csv(PROC / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PROC / "y_test.csv").squeeze()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    # ROC-AUC is undefined when y_true has a single class; scikit-learn either
    # raises or returns NaN depending on version, so handle it explicitly.
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   roc_auc,
    }


def run_grid(name, estimator, param_grid, X_train, y_train, X_test, y_test,
             cv_folds=5, scoring="f1", random_state=42):
    """GridSearchCV + MLflow logging. Returns (best_model, metrics, run_id)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    print(f"\n{'='*55}")
    print(f"  Training: {name}")

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds", cv_folds)

        grid = GridSearchCV(
            estimator, param_grid,
            scoring=scoring, cv=cv, n_jobs=-1, refit=True, verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
        mlflow.log_metric(f"cv_best_{scoring}", round(float(grid.best_score_), 4))

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        # cloudpickle works on MLflow 2.x and 3.x (3.x defaults to skops,
        # which rejects tree models such as RandomForest)
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 serialization_format="cloudpickle")

        print(f"  Best params : {grid.best_params_}")
        print(f"  Test metrics: {metrics}")
        print(classification_report(y_test, y_pred,
              target_names=["No Disease", "Disease"], zero_division=0))

        return best, metrics, run.info.run_id


def train(params: dict = None):
    params = params or load_params()
    data_p = params.get("data", {})
    eval_p = params.get("evaluation", {})
    random_state = data_p.get("random_state", 42)
    cv_folds     = eval_p.get("cv_folds", 5)
    metric       = eval_p.get("primary_metric", "f1")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # MLflow — store runs next to project (override with MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

    X_train, X_test, y_train, y_test = load_data()
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    results = {}
    for name, (estimator, grid) in build_grids(params, spw, random_state).items():
        model, metrics, run_id = run_grid(
            name, estimator, grid, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, scoring=metric, random_state=random_state,
        )
        results[name] = {"model": model, "metrics": metrics, "run_id": run_id}

    # ── Select champion by the primary metric ────────────────────────────
    champion_name = max(results, key=lambda k: results[k]["metrics"][metric])
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
            "path":    path.name,
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
    with open(MODEL_DIR / "champion_info.json", "w") as fh:
        json.dump(champion_info, fh, indent=2)
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
