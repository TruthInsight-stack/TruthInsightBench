#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuro_core import control_metrics, task_data

m = control_metrics(task_data())
x = np.asarray(m["response_mse"])
loo_score = np.asarray([100.0 * (1.0 - (x.sum() - value) / (len(x) - 1)) for value in x])
print(json.dumps({"survive": bool(np.min(loo_score) > 80.0), "value": float(np.min(loo_score)), "family": "sample", "note": "minimum leave-one-trial-out response score (%)"}))
