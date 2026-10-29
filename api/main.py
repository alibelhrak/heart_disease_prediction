"""
FastAPI inference backend.
Runs on port 8000. Called by the Flask frontend.
All paths are resolved relative to this file's location (project root).
"""
import json, joblib, io, math, warnings
import numpy as np
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from src import features as F
from api.schemas import PatientIn

# ── Resolve project root regardless of working directory ──────────────────
ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"

# ── Global state ──────────────────────────────────────────────────────────
MODELS     = {}
MODEL_INFO = {}
SCALER     = None
FEATURES   = []

MODEL_FILES = {
    "LogisticRegression": "logisticregression_model.joblib",
    "RandomForest":       "randomforest_model.joblib",
    "XGBoost":            "xgboost_model.joblib",
}

# Category lists and maps come from src/features.py (shared with training)
BOOL_MAP     = F.BOOL_MAP
CP_VALS      = F.CP_VALS
RESTECG_VALS = F.RESTECG_VALS
SLOPE_VALS   = F.SLOPE_VALS
THAL_VALS    = F.THAL_VALS
DATASET_VALS = F.DATASET_VALS
SEX_VALS     = F.SEX_VALS


def load_artifacts():
    global SCALER, FEATURES, MODEL_INFO
    MODELS.clear()
    SCALER = joblib.load(PROC_DIR / "scaler.joblib")
    with open(PROC_DIR / "feature_names.json") as fh:
        FEATURES = json.load(fh)
    with open(MODEL_DIR / "champion_info.json") as fh:
        MODEL_INFO = json.load(fh)
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            MODELS[name] = joblib.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found")
    print(f"Champion: {MODEL_INFO['champion']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def safe_float(val, default=0.0):
    """Convert to a finite float; anything missing, unparseable, NaN or
    infinite (including the strings "nan" and "inf") becomes `default`."""
    try:
        if val is None:
            return default
        out = float(val)
    except (ValueError, TypeError):
        return default
    return out if math.isfinite(out) else default


def cat_or_default(row: dict, key: str, default: str) -> str:
    """Return a categorical value, falling back to the default when it is
    missing, None, NaN or blank (CSV cells that pandas read as NaN)."""
    val = row.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
        return default
    # Case-insensitive match ("male" -> "Male"); unknown values are passed
    # through unchanged and encode as all zeros.
    return F.canonical_category(key, val) or val


def ohe(val, vals, prefix):
    return {f"{prefix}_{v}": (1 if str(val).strip() == v else 0) for v in vals}


def preprocess_row(row: dict) -> np.ndarray:
    age    = safe_float(row.get("age"), 54)
    thalch = safe_float(row.get("thalch"), 150)
    oldpeak= safe_float(row.get("oldpeak"), 0.0)

    encoded = {
        "age":      age,
        "trestbps": safe_float(row.get("trestbps"), 130),
        "chol":     safe_float(row.get("chol"), 240),
        "thalch":   thalch,
        "oldpeak":  oldpeak,
        "ca":       safe_float(row.get("ca"), 0),
        "fbs":      float(BOOL_MAP.get(row.get("fbs"), 0)),
        "exang":    float(BOOL_MAP.get(row.get("exang"), 0)),
    }
    encoded.update(ohe(cat_or_default(row, "sex", "Male"),             SEX_VALS,     "sex"))
    encoded.update(ohe(cat_or_default(row, "dataset", "Cleveland"),    DATASET_VALS, "dataset"))
    encoded.update(ohe(cat_or_default(row, "cp", "asymptomatic"),      CP_VALS,      "cp"))
    encoded.update(ohe(cat_or_default(row, "restecg", "normal"),       RESTECG_VALS, "restecg"))
    encoded.update(ohe(cat_or_default(row, "slope", "flat"),           SLOPE_VALS,   "slope"))
    encoded.update(ohe(cat_or_default(row, "thal", "normal"),          THAL_VALS,    "thal"))

    # Engineered features: same functions as training (src/features.py)
    encoded["age_group"]            = F.age_group(age)
    encoded["hr_age_ratio"]         = F.hr_age_ratio(thalch, age)
    encoded["st_slope_interaction"] = oldpeak * encoded.get("slope_flat", 0)
    encoded["cp_severe"]            = encoded.get("cp_asymptomatic", 0)

    vec = pd.DataFrame([encoded]).reindex(columns=FEATURES, fill_value=0)
    return SCALER.transform(vec)


def risk_label(prob: float) -> str:
    if prob >= 0.70: return "High"
    if prob >= 0.40: return "Medium"
    return "Low"


def interpret(prob: float) -> str:
    if prob >= 0.70: return "High probability of heart disease. Urgent cardiology referral recommended."
    if prob >= 0.40: return "Moderate risk. Further diagnostic workup advised."
    return "Low probability of heart disease based on provided clinical features."


def row_identifier(d: dict, idx: int) -> int:
    """Patient id from the CSV, or the 1-based row number when the id
    column is absent or the cell is blank."""
    val = d.get("id")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return idx + 1
    return int(val)


def resolve_model(model_name: str):
    """Map "champion" to the champion's name and return (name, model)."""
    if model_name == "champion":
        model_name = MODEL_INFO["champion"]
    if model_name not in MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'. Choose from: {list(MODELS.keys())}")
    return model_name, MODELS[model_name]


def score(model, X) -> tuple:
    """Return (prediction, probability of disease) for one encoded row."""
    return int(model.predict(X)[0]), float(model.predict_proba(X)[0][1])


@app.get("/")
def health():
    return {"status":"ok","champion":MODEL_INFO.get("champion","unknown"),"models":list(MODELS.keys())}


@app.get("/models")
def get_models():
    return {
        "available": list(MODELS.keys()),
        "champion":  MODEL_INFO.get("champion"),
        "all_metrics": {n: r.get("metrics",{}) for n,r in MODEL_INFO.get("all_models",{}).items()},
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), model_name: str = Form("champion")):
    try:
        contents = await file.read()
        df_raw   = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    model_name, model = resolve_model(model_name)
    results = []
    errors  = []

    for idx, row in df_raw.iterrows():
        try:
            d    = row.to_dict()
            X    = preprocess_row(d)
            pred, prob = score(model, X)
            results.append({
                "row_index":              int(idx),
                "id":                     row_identifier(d, idx),
                "age":                    d.get("age"),
                "sex":                    d.get("sex"),
                "dataset":                d.get("dataset"),
                "cp":                     d.get("cp"),
                "trestbps":               d.get("trestbps"),
                "chol":                   d.get("chol"),
                "thalch":                 d.get("thalch"),
                "oldpeak":                d.get("oldpeak"),
                "prediction":             pred,
                "probability_disease":    round(prob, 4),
                "probability_no_disease": round(1-prob, 4),
                "risk_level":             risk_label(prob),
                "interpretation":         interpret(prob),
                "model_used":             model_name,
            })
        except Exception as e:
            errors.append({"row": int(idx), "error": str(e)})

    disease   = sum(1 for r in results if r["prediction"]==1)
    high_risk = sum(1 for r in results if r["risk_level"]=="High")
    metrics   = MODEL_INFO.get("all_models",{}).get(model_name,{}).get("metrics",{})

    return {
        "model_used":       model_name,
        "total_patients":   len(results),
        "disease_count":    disease,
        "no_disease_count": len(results)-disease,
        "high_risk_count":  high_risk,
        "model_metrics":    metrics,
        "results":          results,
        "errors":           errors,
    }


@app.post("/predict")
def predict_one(patient: PatientIn, model_name: str = "champion"):
    """Score a single patient sent as JSON. The body is validated by
    PatientIn (types, ranges, allowed categories); invalid input gets 422."""
    model_name, model = resolve_model(model_name)
    d = patient.model_dump(exclude_none=True)
    pred, prob = score(model, preprocess_row(d))
    return {
        "model_used":             model_name,
        "prediction":             pred,
        "probability_disease":    round(prob, 4),
        "probability_no_disease": round(1 - prob, 4),
        "risk_level":             risk_label(prob),
        "interpretation":         interpret(prob),
        "input":                  d,
    }
