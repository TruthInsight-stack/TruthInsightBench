#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhmc_core import find_structure, read_xyz, task_data, type_z

data = task_data()
deviations = []
for chi, epsilon in ((0.8, 0.04), (0.6, 0.08)):
    _, points = read_xyz(find_structure(data, r1=3, r2=3, chi=chi, epsilon=epsilon))
    z = type_z(points)
    full = (abs(z["C"].mean()) + abs(z["H2"].mean())) / 2
    for label in ("C", "H2"):
        for index in range(len(z[label])):
            trimmed = np.delete(z[label], index)
            centers = {"C": z["C"], "H2": z["H2"]}
            centers[label] = trimmed
            estimate = (abs(centers["C"].mean()) + abs(centers["H2"].mean())) / 2
            deviations.append(abs(float(estimate - full)))
value = max(deviations)
print(json.dumps({"survive": value < 0.05, "value": value, "family": "sample", "note": "maximum layer-center shift after removing any one particle (sigma)"}))
