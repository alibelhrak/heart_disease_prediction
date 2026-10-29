# Testing

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest                                   # all tests
pytest -m "not slow"                     # skip the end-to-end pipeline tests
pytest --cov --cov-branch --cov-report=term-missing   # with branch coverage
```

The suite never reads models or data from the repository. A small synthetic
dataset is pushed through the real `preprocess → train` pipeline once per
session inside a pytest temporary directory (`pipeline_artifacts` fixture in
`tests/conftest.py`), and every test that needs a scaler, feature list or
model uses those artifacts. The FastAPI backend is stubbed with
`requests-mock` for the Flask unit tests; `test_flask_fastapi_live.py` instead
starts the real FastAPI app in a uvicorn server on a free local port.
`test_model_quality.py` trains with the real `params.yaml` grids on the
920-row dataset from `data/raw/generate.py` (about 20 seconds, once per session).

## Layout

```
tests/
├── conftest.py                          shared fixtures (tmp_path artifacts, clients, stubs)
├── unit/
│   ├── test_preprocess_branches.py      impute, encode, engineer          (W5, Ali)
│   ├── test_compute_metrics.py          compute_metrics                   (W5, Ali)
│   ├── test_api_helpers.py              safe_float, ohe, preprocess_row   (W5, Mohaned)
│   ├── test_flask_error_paths.py        Flask /predict error paths        (W5, Mohaned)
│   └── test_features.py                 src/features.py                   (W7, Ali)
├── dataflow/
│   ├── test_defuse_scaler_features.py   scaler + feature list def-use     (W6, Ali)
│   ├── test_train_params.py             params.yaml → train()             (W6, Ali)
│   └── test_defuse_result_store.py      RESULT_STORE def-use + leaks      (W6, Mohaned)
├── integration/
│   ├── test_api_endpoints.py            api/ coverage gaps                (W7, Mohaned)
│   ├── test_flask_routes.py             frontend/ coverage gaps           (W7, Mohaned)
│   ├── test_pipeline_contract.py        preprocess → train → API contract (W9, Ali)
│   ├── test_model_quality.py            quality on the 920-row dataset,
│   │                                    determinism of the pipeline       (W9, Ali)
│   ├── test_csv_validation.py           CSV validation layer + results UI (W9, Mohaned)
│   └── test_flask_fastapi_live.py       Flask ↔ real FastAPI over HTTP    (W9, Mohaned)
└── functional/
    ├── test_f1_gate_bva.py              BVA of the CI F1 gate (0.70)      (W8, Ali)
    ├── test_properties.py               Hypothesis properties; training
    │                                    vs serving consistency            (W8, Ali)
    ├── test_categorical_ep.py           EP of categorical/boolean inputs  (W8, Mohaned)
    ├── test_risk_bva.py                 BVA of risk cut-offs 0.40 / 0.70  (W8, Mohaned)
    ├── test_decision_tables.py          Flask /predict + API /predict/csv (W8, Mohaned)
    └── test_predict_json.py             JSON POST /predict: EP + BVA      (W8, Mohaned)
```

## Defects found by the tests

| Defect | Location | Detected by | Fix |
|---|---|---|---|
| Training/serving skew: API computed `age_group` as `(age-30)//10`, training used bins (0,45], (45,55], (55,65], (65,120] — ages such as 40, 50, 60 got a different feature value at inference | `api/main.py` | `test_api_preprocess_row_matches_training_pipeline`, `test_age_group_parity_at_bin_edges` | Shared `src/features.py` |
| Missing categorical values (NaN) encoded as all-zero vectors in the API instead of a default category | `api/main.py` | `test_preprocess_row_nan_categorical_uses_default` | `cat_or_default()` |
| `RESULT_STORE` grew without bound (memory leak) | `frontend/app.py` | `test_store_is_bounded_across_sessions`, `test_second_predict_replaces_first_in_same_session` | Bounded `OrderedDict`, one entry per session |
| `/export` left a temp file on disk per download | `frontend/app.py` | `test_export_creates_no_temp_files` | Stream CSV from memory |
| Results page crashed (ZeroDivisionError) when every row failed | `results.html` | `test_results_page_with_zero_patients_does_not_crash` | Guard division |
| Training ignored `params.yaml` | `src/train.py` | `test_train_passes_params_to_every_grid` | `load_params()` + `build_grids()` |
| MLflow ≥ 3 refused to log RandomForest (default skops format) | `src/train.py` | `pipeline_artifacts` fixture | `serialization_format="cloudpickle"` |
| `safe_float("nan")` / `safe_float("inf")` returned NaN / inf, so a CSV cell containing the text `nan` made the row's feature vector non-finite and the row failed | `api/main.py` | `test_safe_float_always_finite` (Hypothesis) | Check `math.isfinite` after conversion |
| Categories were case-sensitive: `"male"`, `"FEMALE"`, `"Asymptomatic"` were silently encoded as all zeros (neither category) | `api/main.py` | `test_v2_case_and_whitespace_variants` | `features.canonical_category()` |
| A blank `id` cell made `int(NaN)` fail, so the whole patient was dropped into `errors` | `api/main.py` | `test_blank_id_cell_still_scored` | `row_identifier()` falls back to the row number |
| Invalid CSV values (unknown category such as `Mael`, age 450, cholesterol `high`) were silently replaced by defaults and the patient was scored anyway | `api/main.py` | `test_invalid_rows_are_reported_not_scored` | `api/validation.py`: invalid rows go to `errors`, missing columns to `warnings` |
| Skipped rows were never shown in the web UI | `results.html` | `test_results_page_lists_skipped_rows_and_warnings` | Notices block on the results page |
| When FastAPI rejected a request, Flask showed only "400 Client Error" and dropped the backend's reason | `frontend/app.py` | `test_backend_rejection_is_reported` (live), `test_backend_error_detail_is_shown` | `backend_error()` shows FastAPI's `detail` |
| The F1 gate was a top-level script and could not be tested | `scripts/check_threshold.py` | `test_f1_gate_bva.py` | `passes()` + `main()` |
| ROC-AUC with single-class labels raised or returned NaN depending on scikit-learn version | `src/train.py` | `test_single_class_ground_truth_gives_nan_auc` | Explicit branch |
