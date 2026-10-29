"""
Week 8 (Mohaned) — boundary value analysis for the risk mapping
(api.main.risk_label / interpret), cut-offs at 0.40 and 0.70.

For each boundary b we test b-ε, b and b+ε, where ε is the next representable
float (math.nextafter), plus the domain edges 0.0 and 1.0. The API rounds the
probability it returns to 4 decimals but labels the unrounded value, so we
also check the label against the raw probability end to end.
"""
import math

import pytest

from api.main import interpret, risk_label

MESSAGE_PREFIX = {"Low": "Low probability", "Medium": "Moderate risk",
                  "High": "High probability"}

CASES = [
    (0.0, "Low"),
    (math.nextafter(0.40, 0), "Low"),
    (0.40, "Medium"),
    (math.nextafter(0.40, 1), "Medium"),
    (0.55, "Medium"),
    (math.nextafter(0.70, 0), "Medium"),
    (0.70, "High"),
    (math.nextafter(0.70, 1), "High"),
    (1.0, "High"),
]


@pytest.mark.parametrize("p, label", CASES, ids=[f"{p!r}" for p, _ in CASES])
def test_risk_label_boundaries(p, label):
    assert risk_label(p) == label


@pytest.mark.parametrize("p, label", CASES, ids=[f"{p!r}" for p, _ in CASES])
def test_interpretation_agrees_with_label(p, label):
    assert interpret(p).startswith(MESSAGE_PREFIX[label])


def test_three_classes_cover_the_domain():
    labels = {risk_label(i / 1000) for i in range(1001)}
    assert labels == {"Low", "Medium", "High"}


@pytest.mark.slow
def test_api_labels_match_returned_probabilities(api_client):
    from tests.conftest import csv_upload, make_raw_df
    body = api_client.post("/predict/csv", files=csv_upload(make_raw_df(n=60, seed=11))).json()
    for r in body["results"]:
        p = r["probability_disease"]
        # Rounding to 4 dp can only matter within 5e-5 of a cut-off
        if min(abs(p - 0.40), abs(p - 0.70)) > 5e-5:
            assert r["risk_level"] == risk_label(p)
        assert r["interpretation"] == interpret(p) or min(abs(p - 0.40), abs(p - 0.70)) <= 5e-5
