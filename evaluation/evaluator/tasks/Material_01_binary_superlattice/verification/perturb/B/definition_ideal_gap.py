#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhmc_core import kagome_metrics, task_data

m = kagome_metrics(task_data())
rel = abs(m["mean_layer_separation_sigma"] - m["geometric_ideal_separation_sigma"]) / m["geometric_ideal_separation_sigma"]
print(json.dumps({"survive": rel < 0.05, "value": 100.0 * rel, "family": "definition", "note": "observed-vs-geometric ideal layer-gap relative deviation (%)"}))
