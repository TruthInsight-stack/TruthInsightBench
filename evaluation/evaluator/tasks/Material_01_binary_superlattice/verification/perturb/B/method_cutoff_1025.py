#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhmc_core import kagome_metrics, task_data

m = kagome_metrics(task_data(), cutoff_factor=1.025)
print(json.dumps({"survive": m["large_to_small_contact_degree_median"] == 5.0, "value": m["large_to_small_contact_degree_median"], "family": "method", "note": "tighten contact cutoff from 1.030 to 1.025 times summed radii"}))
