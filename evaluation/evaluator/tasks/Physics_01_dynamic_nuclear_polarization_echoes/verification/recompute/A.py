#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnp_core import summary_a, task_data
print(json.dumps({"values": summary_a(task_data())}))
