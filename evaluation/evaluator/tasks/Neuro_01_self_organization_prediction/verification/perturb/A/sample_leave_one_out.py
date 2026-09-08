#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuro_core import control_metrics, task_data

m = control_metrics(task_data())
x = np.asarray(m["weight_errors"])
loo = np.asarray([(x.sum() - value) / (len(x) - 1) for value in x])
print(json.dumps({"survive": bool(np.max(loo) < 0.04), "value": float(np.max(loo) * 100), "family": "sample", "note": "max leave-one-trial-out session-100 error (%)"}))
