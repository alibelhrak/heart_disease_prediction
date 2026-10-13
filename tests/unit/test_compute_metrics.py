"""Week 5 (Ali) — branch tests for src/train.py: compute_metrics."""
import math


from src.train import compute_metrics

KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_perfect_classifier():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert set(m) == KEYS
    assert all(v == 1.0 for v in m.values())


def test_values_are_python_floats_rounded_to_4dp():
    m = compute_metrics([0, 1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 1],
                        [0.3, 0.7, 0.4, 0.1, 0.9, 0.6, 0.55])
    for v in m.values():
        assert type(v) is float
        assert round(v, 4) == v


def test_zero_division_branch_when_no_positive_predictions():
    # precision is undefined (0/0); zero_division=0 must return 0, not raise
    m = compute_metrics([0, 1, 1], [0, 0, 0], [0.1, 0.4, 0.3])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_inverted_scores_give_zero_auc():
    m = compute_metrics([0, 1], [0, 1], [0.9, 0.1])
    assert m["roc_auc"] == 0.0


def test_single_class_ground_truth_gives_nan_auc():
    # ROC-AUC is undefined with one class; must not raise or warn
    m = compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(m["roc_auc"])
    assert m["accuracy"] == 1.0
