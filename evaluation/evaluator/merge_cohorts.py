#!/usr/bin/env python3
"""Merge independently prepared runs into one validated scoring cohort."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_cohort(path: Path) -> dict:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"cohort root must be a JSON object: {path}")
    if (
        value.get("schema") != "truthinsightbench-scoring-cohort"
        or value.get("release") != "V1.0"
        or not isinstance(value.get("runs"), list)
    ):
        raise SystemExit(f"invalid cohort: {path}")
    return value


def main() -> int:
    args = parse_args()
    runs: list[dict] = []
    for path in args.cohort:
        cohort = load_cohort(path)
        if not all(isinstance(row, dict) for row in cohort["runs"]):
            raise SystemExit(f"cohort contains a non-object run: {path}")
        runs.extend(cohort["runs"])

    required_run_fields = {"key", "agent", "task_id"}
    for row in runs:
        missing = sorted(required_run_fields - row.keys())
        if missing:
            raise SystemExit(f"cohort run is missing fields: {', '.join(missing)}")
        if not all(isinstance(row[field], str) and row[field] for field in required_run_fields):
            raise SystemExit("cohort run identity fields must be non-empty strings")
    if not runs:
        raise SystemExit("cohort contains no runs")

    keys = [row["key"] for row in runs]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate run key")
    pairs = [(row["task_id"], row["agent"]) for row in runs]
    if len(pairs) != len(set(pairs)):
        raise SystemExit("duplicate task-and-agent run")
    for row in runs:
        expected_key = f"{row['agent']}--{row['task_id']}"
        if row["key"] != expected_key:
            raise SystemExit(f"run key does not match run identity: {row['key']}")

    agents = sorted({row["agent"] for row in runs})
    task_agents: dict[str, set[str]] = defaultdict(set)
    for row in runs:
        task_agents[row["task_id"]].add(row["agent"])
    complete = sorted(
        task for task, present_agents in task_agents.items() if present_agents == set(agents)
    )
    provisional = sorted(set(task_agents) - set(complete))
    summary = {
        "run_count": len(runs),
        "complete_task_count": len(complete),
        "complete_tasks": complete,
        "provisional_task_count": len(provisional),
        "provisional_tasks": provisional,
    }
    value = {
        "schema": "truthinsightbench-scoring-cohort",
        "release": "V1.0",
        "agents": agents,
        "summary": summary,
        "runs": sorted(runs, key=lambda row: (row["task_id"], row["agent"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "READY", "output": str(args.output.resolve()), **summary},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
