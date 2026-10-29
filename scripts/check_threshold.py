"""CI gate: fail if the champion's F1 is below params.yaml evaluation.min_f1_threshold."""
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INFO_FILE = ROOT / "models" / "champion_info.json"
PARAMS_FILE = ROOT / "params.yaml"


def passes(f1: float, threshold: float) -> bool:
    """The gate is inclusive: F1 equal to the threshold passes; NaN fails."""
    return not math.isnan(f1) and f1 >= threshold


def main(info_path: Path = INFO_FILE, params_path: Path = PARAMS_FILE) -> int:
    info = json.loads(Path(info_path).read_text())
    threshold = yaml.safe_load(Path(params_path).read_text())["evaluation"]["min_f1_threshold"]
    f1 = float(info["metrics"]["f1"])
    print(f"Champion: {info['champion']}  F1={f1}  threshold={threshold}")
    if not passes(f1, threshold):
        print(f"FAIL: F1 below {threshold}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
