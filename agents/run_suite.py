#!/usr/bin/env python3
"""Run selected tasks for one or all published Agent profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


AGENTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENTS_ROOT.parent


def main() -> int:
    profiles = json.loads((AGENTS_ROOT / "agent_profiles.json").read_text(encoding="utf-8"))
    registry = json.loads((REPOSITORY_ROOT / "registry.json").read_text(encoding="utf-8"))
    agent_choices = sorted(profiles["agents"])
    task_choices = [row["task_id"] for row in registry["tasks"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="append", choices=agent_choices, help="repeatable; defaults to all four")
    parser.add_argument("--task-id", action="append", choices=task_choices, help="repeatable; defaults to all 40")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--model", default=profiles["default_participant_model"])
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--container-image")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the resolved matrix without running it")
    args = parser.parse_args()

    agents = args.agent or agent_choices
    tasks = args.task_id or task_choices
    matrix = [{"agent": agent, "task_id": task} for agent in agents for task in tasks]
    if args.list:
        print(json.dumps({"run_count": len(matrix), "runs": matrix}, indent=2))
        return 0

    failures = []
    for row in matrix:
        run_root = args.runs_root.resolve() / row["agent"] / row["task_id"]
        command = [
            sys.executable,
            str(AGENTS_ROOT / "run_agent.py"),
            "--agent",
            row["agent"],
            "--task-id",
            row["task_id"],
            "--run-root",
            str(run_root),
            "--model",
            args.model,
        ]
        if args.timeout_seconds is not None:
            command.extend(["--timeout-seconds", str(args.timeout_seconds)])
        if args.container_image:
            command.extend(["--container-image", args.container_image])
        completed = subprocess.run(command)
        if completed.returncode:
            failures.append({**row, "exit_code": completed.returncode})
            if not args.continue_on_error:
                break
    print(
        json.dumps(
            {"status": "PASS" if not failures else "FAIL", "scheduled": len(matrix), "failures": failures},
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
