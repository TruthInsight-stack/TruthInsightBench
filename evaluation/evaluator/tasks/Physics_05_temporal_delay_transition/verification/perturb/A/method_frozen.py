#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from science_core import perturb, task_data

print(json.dumps(perturb("A", "method", task_data()), ensure_ascii=False))
