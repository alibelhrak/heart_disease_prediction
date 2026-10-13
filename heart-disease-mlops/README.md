# Heart Disease MLOps Pipeline

End-to-end MLOps project: UCI Heart Disease dataset → 3 trained models → FastAPI backend → Flask web UI → Docker → Kubernetes → GitHub Actions CI/CD.

![Demo](https://github.com/user-attachments/assets/f9272ff5-1cc7-49db-afa6-aa962db5c018)


## Dataset

This project uses the **UCI Heart Disease (Cleveland) dataset**, publicly available at:
🔗 https://archive.ics.uci.edu/ml/datasets/heart+Disease

The dataset originates from the landmark study by Detrano et al., *"International application of a new probability algorithm for the diagnosis of coronary artery disease"* (cited 710+ times). It contains **303 patient records** described by **13 clinical features**:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) |

**Target:** Binary — disease present (1) vs. absent (0).

---

## Project Report

A full written report is included in this repository: [`mlops_heart_disease_report.pdf`](./mlops_heart_disease_report.pdf)

The report (3 pages, LaTeX-typeset) documents the entire pipeline and covers:

- **Problem Description** — dataset background, clinical features, preprocessing decisions, and why F1-score was chosen as the champion metric over accuracy
- **System Architecture** — a detailed 7-layer pipeline diagram (data versioning → training → experiment tracking → REST serving → web frontend → containerisation → CI/CD automation), plus a breakdown table of every layer's technology and artefacts
- **Tools Used and Justification** — reasoning behind each technology choice: DVC, MLflow, scikit-learn, XGBoost, FastAPI, Flask, Docker, Kubernetes, and GitHub Actions
- **Challenges Faced** — real bugs encountered during development with root causes and fixes (missing `python-multipart`, Flask cookie overflow, relative path resolution, Minikube image pull, and scaler data leakage)

> **Authors:** Ali Belhrak & Rahma Aroua

---

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc init

# 2. Generate / place dataset
python data/raw/generate.py          # or drop heart_disease_uci.csv into data/raw/

# 3. Run full pipeline
python src/preprocess.py
python src/train.py                  # trains LR + RF + XGBoost, picks champion

# 4. Start FastAPI backend (port 8000)
uvicorn api.main:app --reload --port 8000

# 5. Start Flask frontend (port 5000) — new terminal
cd frontend && python app.py

# Open http://localhost:5000
```

---

## Docker

```bash
docker build -t heart-disease-app .
docker run -p 8000:8000 -p 5000:5000 heart-disease-app
# UI  → http://localhost:5000
# API → http://localhost:8000/docs
```

---

## Kubernetes (Minikube)

```bash
minikube start
# Edit k8s/deployment.yaml → replace YOUR_DOCKERHUB_USERNAME
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods                          # wait for Running
minikube service heart-disease-app        # opens UI in browser
```

---

## GitHub Actions Secrets

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## Models

| Model | F1 | ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.838 | 0.924 | Baseline, interpretable |
| Random Forest | 0.800 | 0.893 | Ensemble |
| XGBoost ★ | 0.843 | 0.906 | Champion — best F1 |

---

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── raw/heart_disease_uci.csv          ← DVC tracked
│   └── processed/                         ← DVC outputs
├── src/
│   ├── preprocess.py                      ← DVC stage 1
│   └── train.py                           ← DVC stage 2
├── api/
│   └── main.py                            ← FastAPI (port 8000)
├── frontend/
│   ├── app.py                             ← Flask (port 5000)
│   └── templates/
│       ├── upload.html                    ← Screen 1: upload + model select
│       └── results.html                   ← Screen 2: predictions table
├── models/                                ← trained model artifacts
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/
│   └── pipeline.yaml                      ← CI/CD
├── mlops_heart_disease_report.pdf         ← Project report
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── start.sh
└── requirements.txt
```

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/models` | List models + metrics |
| POST | `/predict/csv` | Upload CSV → JSON predictions |

---

## Web App Flow

1. Open `http://localhost:5000`
2. Upload a CSV (same schema as `heart_disease_uci.csv`)
3. Choose one of 3 models
4. Click **Run predictions** → redirected to results page
5. Filter by disease / no disease / high risk
6. Sort any column
7. Export results as CSV
