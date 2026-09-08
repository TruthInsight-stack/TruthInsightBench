#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhmc_core import find_structure, read_xyz, task_data

data = task_data(); estimates = []
for chi, epsilon in ((0.8, 0.04), (0.6, 0.08)):
    _, points = read_xyz(find_structure(data, r1=3, r2=3, chi=chi, epsilon=epsilon))
    z = np.asarray([point[3] for point in points])
    estimates.append((abs(z[z < 0].mean()) + abs(z[z > 0].mean())) / 2)
survive = abs(estimates[0] - 2.4912322222) < 0.02 and abs(estimates[1] - 2.0977411111) < 0.02
print(json.dumps({"survive": bool(survive), "value": float(max(estimates)), "family": "definition", "note": "define layers by z sign rather than particle type"}))
