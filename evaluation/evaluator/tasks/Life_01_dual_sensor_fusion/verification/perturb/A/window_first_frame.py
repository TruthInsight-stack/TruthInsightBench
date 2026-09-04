#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from syt_core import first_frame_runs, task_data

r = first_frame_runs(task_data())
drop = float(r["None"].mean() - r["1:200 SensorB"].mean())
print(json.dumps({"survive": drop > 35.0, "value": drop, "family": "window", "note": "first frame rather than first two frames; drop in percentage points"}))
