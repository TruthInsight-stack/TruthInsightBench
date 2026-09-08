#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from syt_core import binding_metrics, task_data

m = binding_metrics(task_data(), reducer=np.median)
print(json.dumps({"survive": m["binding_drop_pct"] > 80.0, "value": m["binding_drop_pct"], "family": "method", "note": "median rather than mean across runs"}))
