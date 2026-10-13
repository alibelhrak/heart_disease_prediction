"""CI gate: fail if the champion's F1 is below params.yaml evaluation.min_f1_threshold."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

info = json.loads((ROOT / "models" / "champion_info.json").read_text())
threshold = yaml.safe_load((ROOT / "params.yaml").read_text())["evaluation"]["min_f1_threshold"]
f1 = info["metrics"]["f1"]
print(f"Champion: {info['champion']}  F1={f1}  threshold={threshold}")
if f1 < threshold:
    print(f"FAIL: F1 below {threshold}")
    sys.exit(1)
print("PASS")
