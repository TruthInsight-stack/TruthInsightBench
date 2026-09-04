#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

here = Path(__file__).resolve()
runtime = next(parent for parent in here.parents if (parent / "verification_runtime/scientific_actions_tabular.py").is_file())
sys.path.insert(0, str(runtime / "verification_runtime"))
from scientific_actions_tabular import run

data = Path(os.environ["TASK_DATA"]).resolve()
print(json.dumps(run("Material_03_membrane_selective_permeation", data, "B", "sample"), ensure_ascii=False, sort_keys=True))
