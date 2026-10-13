"""
FastAPI inference backend.
Runs on port 8000. Called by the Flask frontend.
All paths are resolved relative to this file's location (project root).
"""
import json, joblib, io, warnings
import numpy as np
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

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

BOOL_MAP     = {True:1,False:0,"True":1,"False":0,1:1,0:0,"true":1,"false":0,"1":1,"0":0}
CP_VALS      = ["typical angina","atypical angina","non-anginal","asymptomatic"]
RESTECG_VALS = ["normal","st-t abnormality","lv hypertrophy"]
SLOPE_VALS   = ["upsloping","flat","downsloping"]
THAL_VALS    = ["normal","fixed defect","reversable defect"]
DATASET_VALS = ["Cleveland","Hungarian","Switzerland","VA Long Beach"]
SEX_VALS     = ["Male","Female"]


def load_artifacts():
    global SCALER, FEATURES, MODEL_INFO
    SCALER     = joblib.load(PROC_DIR / "scaler.joblib")
    FEATURES   = json.load(open(PROC_DIR / "feature_names.json"))
    MODEL_INFO = json.load(open(MODEL_DIR / "champion_info.json"))
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
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


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
    encoded.update(ohe(row.get("sex","Male"),            SEX_VALS,     "sex"))
    encoded.update(ohe(row.get("dataset","Cleveland"),   DATASET_VALS, "dataset"))
    encoded.update(ohe(row.get("cp","asymptomatic"),     CP_VALS,      "cp"))
    encoded.update(ohe(row.get("restecg","normal"),      RESTECG_VALS, "restecg"))
    encoded.update(ohe(row.get("slope","flat"),          SLOPE_VALS,   "slope"))
    encoded.update(ohe(row.get("thal","normal"),         THAL_VALS,    "thal"))

    encoded["age_group"]            = int(min(max((age - 30) // 10, 0), 3))
    encoded["hr_age_ratio"]         = thalch / age if age > 0 else 0
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

    if model_name == "champion":
        model_name = MODEL_INFO["champion"]
    if model_name not in MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'. Choose from: {list(MODELS.keys())}")

    model   = MODELS[model_name]
    results = []
    errors  = []

    for idx, row in df_raw.iterrows():
        try:
            d    = row.to_dict()
            X    = preprocess_row(d)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            results.append({
                "row_index":              int(idx),
                "id":                     int(d.get("id", idx+1)),
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
