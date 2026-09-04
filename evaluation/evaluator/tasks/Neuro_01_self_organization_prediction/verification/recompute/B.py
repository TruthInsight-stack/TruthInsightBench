#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuro_core import control_metrics, task_data

metrics = control_metrics(task_data())
print(json.dumps({"values": {
    "initial_sessions": metrics["ninit_sessions"],
    "forecast_end_session": 100,
    "independent_trials": metrics["n_trials"],
    "session100_response_score_pct": metrics["session100_response_score_pct"],
}}))
