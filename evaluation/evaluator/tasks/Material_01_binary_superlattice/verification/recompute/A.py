#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bhmc_core import bilayer_metrics, task_data

print(json.dumps({"values": bilayer_metrics(task_data())}))
