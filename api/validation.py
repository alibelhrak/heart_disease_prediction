"""
CSV validation layer for POST /predict/csv.

Before this layer existed, every value the API could not use was silently
replaced by a hard-coded default: a typo such as "Mael", an age of 450 or a
cholesterol of "high" was scored as if it were a normal patient.

validate_frame() now checks an uploaded CSV before any row is scored:

  * file level  - at least one expected feature column must be present
                  (otherwise CSVValidationError → HTTP 400); expected columns
                  that are missing and unknown extra columns become warnings
  * row level   - numbers must parse, be finite and lie in
                  features.NUMERIC_RANGES; categories must be known (any case);
                  booleans must be true/false/1/0; `ca` and `id` must be whole
                  numbers. A row with any problem is reported in `errors`
                  and is not scored.

Blank cells are still allowed and are imputed, exactly as in training.
The ranges and categories are the ones the JSON schema uses (src/features.py).
"""
import math

import numpy as np
import pandas as pd

from src import features as F

# Values the API substitutes for a missing column/cell (see preprocess_row)
DEFAULTS = {"age": 54, "trestbps": 130, "chol": 240, "thalch": 150, "oldpeak": 0.0,
            "ca": 0, "fbs": False, "exang": False, "sex": "Male", "dataset": "Cleveland",
            "cp": "asymptomatic", "restecg": "normal", "slope": "flat", "thal": "normal"}
FEATURE_COLUMNS = F.NUM_COLS + F.BOOL_COLS + F.CAT_COLS
PASSTHROUGH = {"id", "num"}            # allowed, not features
INTEGER_COLS = {"ca", "id"}


class CSVValidationError(ValueError):
    """The file as a whole cannot be used."""


def _blank(val) -> bool:
    return val is None or (isinstance(val, float) and np.isnan(val)) or \
        (isinstance(val, str) and val.strip() == "")


def _fmt(x: float) -> str:
    return f"{x:g}"


def check_number(col: str, val):
    """Return an error message, or None if the value is acceptable."""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return f"{col}: '{val}' is not a number"
    if not math.isfinite(x):
        return f"{col}: '{val}' is not a finite number"
    if col in INTEGER_COLS and x != int(x):
        return f"{col}: {_fmt(x)} is not a whole number"
    if col in F.NUMERIC_RANGES:
        lo, hi = F.NUMERIC_RANGES[col]
        if not lo <= x <= hi:
            return f"{col}: {_fmt(x)} is outside the allowed range {_fmt(lo)}–{_fmt(hi)}"
    return None


def check_category(col: str, val):
    if F.canonical_category(col, val) is None:
        return f"{col}: '{val}' is not one of {', '.join(F.CATEGORIES[col])}"
    return None


def check_bool(col: str, val):
    key = val.strip() if isinstance(val, str) else val
    try:
        known = key in F.BOOL_MAP
    except TypeError:                     # unhashable
        known = False
    if not known:
        return f"{col}: '{val}' is not true/false"
    return None


CHECKS = {**{c: check_number for c in F.NUM_COLS + ["id"]},
          **{c: check_category for c in F.CAT_COLS},
          **{c: check_bool for c in F.BOOL_COLS}}


def check_row(row: dict) -> list:
    """All problems in one row, in column order."""
    problems = []
    for col, check in CHECKS.items():
        val = row.get(col)
        if col in row and not _blank(val):
            msg = check(col, val)
            if msg:
                problems.append(msg)
    return problems


def validate_frame(df: pd.DataFrame):
    """Return (row_errors, warnings).

    row_errors: {row_index: [messages]} for rows that must not be scored.
    warnings:   file-level messages (missing / ignored columns).
    """
    cols = [str(c).strip() for c in df.columns]
    present = [c for c in FEATURE_COLUMNS if c in cols]
    if not present:
        raise CSVValidationError(
            "CSV has none of the expected columns: " + ", ".join(FEATURE_COLUMNS))

    warnings = [f"Column '{c}' is missing; the default {DEFAULTS[c]!r} is used for every patient."
                for c in FEATURE_COLUMNS if c not in cols]
    warnings += [f"Column '{c}' is not used by the model and was ignored."
                 for c in cols if c not in FEATURE_COLUMNS and c not in PASSTHROUGH]

    row_errors = {}
    for idx, row in enumerate(df.to_dict("records")):
        problems = check_row({str(k).strip(): v for k, v in row.items()})
        if problems:
            row_errors[idx] = problems
    return row_errors, warnings
