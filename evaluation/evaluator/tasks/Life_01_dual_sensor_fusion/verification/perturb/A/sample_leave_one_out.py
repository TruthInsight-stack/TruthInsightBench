#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from syt_core import fusion_runs, task_data

r = fusion_runs(task_data())
tn, th = r[("None", "total_fusion")], r[("1:200 SensorB", "total_fusion")]
en, eh = r[("None", "fusion_first_two_frames")], r[("1:200 SensorB", "fusion_first_two_frames")]
def loo(x):
    return np.asarray([(x.sum() - value) / (len(x) - 1) for value in x])
total_change = loo(th) - loo(tn)
early_drop = loo(en) - loo(eh)
survive = bool(np.max(np.abs(total_change)) < 5.0 and np.min(early_drop) > 45.0)
print(json.dumps({"survive": survive, "value": float(np.min(early_drop)), "family": "sample", "note": "paired-index leave-one-run-out; min early drop (pp)"}))
