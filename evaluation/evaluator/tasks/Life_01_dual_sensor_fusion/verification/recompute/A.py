#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syt_core import fusion_metrics, task_data

print(json.dumps({"values": fusion_metrics(task_data())}))
