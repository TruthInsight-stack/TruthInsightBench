#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from syt_core import binding_runs, task_data

r = binding_runs(task_data())
zero, high = r[0.0], r[30.0]
def loo(x):
    return np.asarray([(x.sum() - value) / (len(x) - 1) for value in x])
drops = 100.0 * (loo(zero) - loo(high)) / loo(zero)
print(json.dumps({"survive": bool(np.min(drops) > 80.0), "value": float(np.min(drops)), "family": "sample", "note": "paired-index leave-one-run-out; minimum inhibition (%)"}))
