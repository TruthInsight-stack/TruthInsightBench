#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from syt_core import binding_metrics, task_data

m = binding_metrics(task_data(), high_concentration=20.0)
print(json.dumps({"survive": m["binding_drop_pct"] > 80.0, "value": m["binding_drop_pct"], "family": "definition", "note": "use 20 uM as the high-competition threshold"}))
