"""
Week 8 (Ali) — boundary value analysis for the CI model-quality gate
(scripts/check_threshold.py). The gate fails the pipeline when the champion's
F1 is below params.yaml evaluation.min_f1_threshold (0.70).

  Partition          F1 values tested              Expected
  -----------------  ----------------------------  --------
  below threshold    0.0, 0.6999, nextafter(0.70,0) FAIL
  on threshold       0.70                          PASS (inclusive)
  above threshold    nextafter(0.70,1), 0.7001, 1.0 PASS
  invalid            NaN                           FAIL
"""
import json
import math

import pytest
import yaml

from scripts import check_threshold as gate

T = 0.70


@pytest.mark.parametrize("f1, ok", [
    (0.0, False),
    (0.6999, False),
    (math.nextafter(T, 0), False),      # just below
    (T, True),                          # on the boundary
    (math.nextafter(T, 1), True),       # just above
    (0.7001, True),
    (1.0, True),
    (float("nan"), False),
])
def test_passes_boundaries(f1, ok):
    assert gate.passes(f1, T) is ok


@pytest.fixture
def write_files(tmp_path):
    def _write(f1, threshold=T):
        info = tmp_path / "champion_info.json"
        params = tmp_path / "params.yaml"
        info.write_text(json.dumps({"champion": "LogisticRegression",
                                    "metrics": {"f1": f1}}))
        params.write_text(yaml.safe_dump({"evaluation": {"min_f1_threshold": threshold}}))
        return info, params
    return _write


@pytest.mark.parametrize("f1, code", [(0.6999, 1), (0.70, 0), (0.7001, 0)])
def test_main_exit_code(write_files, capsys, f1, code):
    assert gate.main(*write_files(f1)) == code
    out = capsys.readouterr().out
    assert ("PASS" in out) is (code == 0)
    assert ("FAIL" in out) is (code == 1)


def test_threshold_comes_from_params(write_files):
    # Same F1, different thresholds → different verdicts
    assert gate.main(*write_files(0.75, threshold=0.70)) == 0
    assert gate.main(*write_files(0.75, threshold=0.80)) == 1


def test_repository_params_define_the_gate():
    params = yaml.safe_load(gate.PARAMS_FILE.read_text())
    assert params["evaluation"]["min_f1_threshold"] == pytest.approx(T)


def test_missing_champion_file_is_an_error(tmp_path, write_files):
    _, params = write_files(0.8)
    with pytest.raises(FileNotFoundError):
        gate.main(tmp_path / "nope.json", params)
