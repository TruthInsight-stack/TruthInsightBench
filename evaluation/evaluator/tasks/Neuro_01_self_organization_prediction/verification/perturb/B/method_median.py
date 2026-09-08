#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuro_core import control_metrics, task_data

m = control_metrics(task_data())
value = m["session100_response_score_median_pct"]
print(json.dumps({"survive": value > 80.0, "value": value, "family": "method", "note": "median rather than mean across independent trials"}))
