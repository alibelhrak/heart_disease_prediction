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
`requests-mock` for all Flask tests.

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
└── integration/
    ├── test_api_endpoints.py            api/ coverage gaps                (W7, Mohaned)
    └── test_flask_routes.py             frontend/ coverage gaps           (W7, Mohaned)
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
| ROC-AUC with single-class labels raised or returned NaN depending on scikit-learn version | `src/train.py` | `test_single_class_ground_truth_gives_nan_auc` | Explicit branch |
