#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from syt_core import fusion_metrics, task_data

m = fusion_metrics(task_data(), reducer=np.median)
survive = abs(m["total_fusion_change_pp"]) < 5.0 and m["early_fusion_drop_pp"] > 45.0
print(json.dumps({"survive": bool(survive), "value": m["early_fusion_drop_pp"], "family": "method", "note": "median rather than mean across runs"}))
