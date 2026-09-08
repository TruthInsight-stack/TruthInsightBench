#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuro_core import control_metrics, task_data

m = control_metrics(task_data())
value = 100.0 * m["session100_weight_error_median"]
print(json.dumps({"survive": value < 4.0, "value": value, "family": "method", "note": "median across independent trials"}))
