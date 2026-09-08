#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuro_core import control_metrics, task_data

m = control_metrics(task_data(), ninit=8)
value = 100.0 * m["session100_weight_error_mean"]
print(json.dumps({"survive": value < 6.0, "value": value, "family": "window", "note": "use first 8 rather than 10 sessions"}))
