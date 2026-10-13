"""
Flask frontend — runs on port 5000.
Talks to FastAPI on port 8000.

FIX: Results stored server-side (temp file) instead of Flask cookie session,
     because 920 rows of JSON (~45KB) far exceeds the 4KB cookie limit.
"""
import os, json, uuid, tempfile, requests
from pathlib import Path
from flask import (Flask, render_template, request,
                   redirect, url_for, session, jsonify, send_file)

# ── Resolve template/static folders from this file's location ────────────
HERE = Path(__file__).resolve().parent
app  = Flask(__name__, template_folder=str(HERE / "templates"),
                        static_folder=str(HERE / "static"))
app.secret_key = "heart-mlops-secret-2024"

# Server-side result store (temp dir, survives the request/response cycle)
RESULT_STORE = {}

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
    result_id = str(uuid.uuid4())
    RESULT_STORE[result_id] = {
        "data":     data,
        "model":    model_name,
        "filename": f.filename,
    }
    session["result_id"] = result_id
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

    cols = ["id","age","sex","dataset","cp","trestbps","chol","thalch",
            "oldpeak","prediction","probability_disease","probability_no_disease",
            "risk_level","interpretation","model_used"]

    lines = [",".join(cols)]
    for r in results:
        row = []
        for c in cols:
            v = r.get(c, "")
            s = "" if v is None else str(v)
            row.append(f'"{s}"' if "," in s else s)
        lines.append(",".join(row))

    csv_bytes = "\n".join(lines).encode("utf-8")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
    tmp.write(csv_bytes)
    tmp.close()

    return send_file(tmp.name, as_attachment=True,
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
