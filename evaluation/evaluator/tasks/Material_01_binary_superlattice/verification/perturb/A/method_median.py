#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhmc_core import bilayer_metrics, task_data

m = bilayer_metrics(task_data(), reducer=np.median)
survive = 2.35 < m["hex_abs_z_sigma"] < 2.65 and 1.9 < m["square_abs_z_sigma"] < 2.3
print(json.dumps({"survive": bool(survive), "value": m["hex_abs_z_sigma"] - m["square_abs_z_sigma"], "family": "method", "note": "median rather than mean layer center; hex-minus-square separation"}))
