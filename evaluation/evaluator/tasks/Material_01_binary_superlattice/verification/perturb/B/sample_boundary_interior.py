#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhmc_core import kagome_metrics, task_data

m = kagome_metrics(task_data())
interior = [degree for degree in m["large_to_small_contact_degrees"] if degree == max(m["large_to_small_contact_degrees"])]
survive = len(interior) >= 2 and min(interior) == 6
print(json.dumps({"survive": survive, "value": len(interior), "family": "sample", "note": "remove boundary large particles (degree 4); retained interior particles each have six cross-type contacts"}))
