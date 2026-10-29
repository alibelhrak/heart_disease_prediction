"""
Request schema for the single-patient JSON endpoint POST /predict.

Allowed categories and numeric ranges come from src/features.py, so the
JSON endpoint and the CSV validation layer accept exactly the same values.
Required fields: age, sex, cp. Every other field is optional and falls back
to the same default the CSV endpoint uses.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import features as F


def _range(name: str, required: bool = False):
    lo, hi = F.NUMERIC_RANGES[name]
    return Field(... if required else None, ge=lo, le=hi, allow_inf_nan=False)


Sex     = Literal[tuple(F.SEX_VALS)]
Dataset = Literal[tuple(F.DATASET_VALS)]
Cp      = Literal[tuple(F.CP_VALS)]
Restecg = Literal[tuple(F.RESTECG_VALS)]
Slope   = Literal[tuple(F.SLOPE_VALS)]
Thal    = Literal[tuple(F.THAL_VALS)]


class PatientIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age:      float             = _range("age", required=True)
    sex:      Sex
    cp:       Cp
    dataset:  Optional[Dataset] = None
    trestbps: Optional[float]   = _range("trestbps")
    chol:     Optional[float]   = _range("chol")
    fbs:      Optional[bool]    = None
    restecg:  Optional[Restecg] = None
    thalch:   Optional[float]   = _range("thalch")
    exang:    Optional[bool]    = None
    oldpeak:  Optional[float]   = _range("oldpeak")
    slope:    Optional[Slope]   = None
    ca:       Optional[int]     = _range("ca")
    thal:     Optional[Thal]    = None

    @field_validator(*F.CAT_COLS, mode="before")
    @classmethod
    def _canonical(cls, v, info):
        """Accept any case/whitespace ("male", " Male ") for categories."""
        if v is None:
            return v
        return F.canonical_category(info.field_name, v) or v
