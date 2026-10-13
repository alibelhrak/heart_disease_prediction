"""
Flask frontend — runs on port 5000.
Talks to FastAPI on port 8000.

Results are stored server-side (in RESULT_STORE) instead of in the Flask
cookie session, because 920 rows of JSON (~45KB) far exceeds the 4KB limit.

RESULT_STORE is bounded (oldest entries are evicted) and each session keeps
at most one entry, so memory does not grow without limit. /export streams
the CSV from memory, so no temporary files are left on disk.
"""
import os, io, csv, uuid, requests
from collections import OrderedDict
from pathlib import Path
from flask import (Flask, render_template, request,
                   redirect, url_for, session, jsonify, send_file)

# ── Resolve template/static folders from this file's location ────────────
HERE = Path(__file__).resolve().parent
app  = Flask(__name__, template_folder=str(HERE / "templates"),
                        static_folder=str(HERE / "static"))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "heart-mlops-secret-2024")

# Server-side result store: insertion-ordered so the oldest can be evicted
RESULT_STORE: "OrderedDict[str, dict]" = OrderedDict()
MAX_STORED_RESULTS = int(os.environ.get("MAX_STORED_RESULTS", "50"))

EXPORT_COLS = ["id","age","sex","dataset","cp","trestbps","chol","thalch",
               "oldpeak","prediction","probability_disease","probability_no_disease",
               "risk_level","interpretation","model_used"]


def store_result(entry: dict) -> str:
    """Save a result, replacing this session's previous one and evicting the
    oldest entries once MAX_STORED_RESULTS is exceeded. Returns the new id."""
    old_id = session.get("result_id")
    if old_id is not None:
        RESULT_STORE.pop(old_id, None)
    result_id = str(uuid.uuid4())
    RESULT_STORE[result_id] = entry
    while len(RESULT_STORE) > MAX_STORED_RESULTS:
        RESULT_STORE.popitem(last=False)
    session["result_id"] = result_id
    return result_id


def results_to_csv(results: list) -> bytes:
    """Serialise prediction rows to CSV with proper quoting."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(EXPORT_COLS)
    for r in results:
        writer.writerow(["" if r.get(c) is None else r.get(c) for c in EXPORT_COLS])
    return buf.getvalue().encode("utf-8")

FASTAPI_URL  = os.environ.get("FASTAPI_URL", "http://localhost:8000")


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("upload.html", error="No file uploaded.")

    f          = request.files["file"]
    model_name = request.form.get("model", "champion")

    if f.filename == "":
        return render_template("upload.html", error="Please select a CSV file.")

    if not f.filename.lower().endswith(".csv"):
        return render_template("upload.html", error="Only CSV files are accepted.")

    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict/csv",
            files={"file": (f.filename, f.read(), "text/csv")},
            data={"model_name": model_name},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        return render_template("upload.html",
            error=f"Cannot reach FastAPI backend at {FASTAPI_URL}. "
                   "Make sure it is running: uvicorn api.main:app --port 8000")
    except requests.exceptions.Timeout:
        return render_template("upload.html",
            error="Request timed out. The file may be too large.")
    except Exception as e:
        return render_template("upload.html", error=f"Prediction failed: {e}")

    # Store results server-side (avoids 4KB cookie overflow)
    store_result({"data": data, "model": model_name, "filename": f.filename})
    return redirect(url_for("results"))


@app.route("/results")
def results():
    result_id = session.get("result_id")
    if not result_id or result_id not in RESULT_STORE:
        return redirect(url_for("index"))

    stored   = RESULT_STORE[result_id]
    data     = stored["data"]
    model    = stored["model"]
    filename = stored["filename"]
    return render_template("results.html", data=data, model=model, filename=filename)


@app.route("/export")
def export():
    """Download predictions as CSV."""
    result_id = session.get("result_id")
    if not result_id or result_id not in RESULT_STORE:
        return redirect(url_for("index"))

    results = RESULT_STORE[result_id]["data"].get("results", [])
    if not results:
        return redirect(url_for("results"))

    return send_file(io.BytesIO(results_to_csv(results)), as_attachment=True,
                     download_name="heart_disease_predictions.csv",
                     mimetype="text/csv")


@app.route("/api/models")
def api_models():
    """Proxy to FastAPI /models for the frontend JS."""
    try:
        r = requests.get(f"{FASTAPI_URL}/models", timeout=5)
        return jsonify(r.json())
    except Exception:
        return jsonify({
            "available": ["LogisticRegression","RandomForest","XGBoost"],
            "champion":  "XGBoost",
        })


@app.route("/health")
def health():
    try:
        r = requests.get(f"{FASTAPI_URL}/", timeout=3)
        api_ok = r.status_code == 200
    except Exception:
        api_ok = False
    return jsonify({"flask":"ok","fastapi":"ok" if api_ok else "unreachable"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
