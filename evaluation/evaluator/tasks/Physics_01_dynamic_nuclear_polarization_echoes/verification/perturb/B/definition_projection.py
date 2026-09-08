#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dnp_core import perturb, task_data
print(json.dumps(perturb("B", "definition", task_data())))
