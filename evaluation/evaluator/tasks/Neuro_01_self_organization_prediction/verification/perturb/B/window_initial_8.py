#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuro_core import control_metrics, task_data

m = control_metrics(task_data(), ninit=8)
value = m["session100_response_score_pct"]
print(json.dumps({"survive": value > 75.0, "value": value, "family": "window", "note": "use first 8 rather than 10 sessions"}))
