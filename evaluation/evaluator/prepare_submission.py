#!/usr/bin/env python3
"""Validate one frozen Agent run and prepare a scoring cohort entry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
AGENTS_ROOT = REPOSITORY_ROOT / "agents"
sys.path.insert(0, str(AGENTS_ROOT))

from workspace_integrity import sha256_file, tree_snapshot, validate_workspace  # noqa: E402


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


def lexical_absolute(path: Path, *, base: Path | None = None) -> Path:
    candidate = path if path.is_absolute() else (base / path if base else path)
    return Path(os.path.abspath(candidate))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def receipt_path(
    receipt: dict,
    field: str,
    *,
    receipt_directory: Path,
) -> Path:
    value = receipt.get(field)
    require(isinstance(value, str) and bool(value.strip()), f"missing run receipt field: {field}")
    return lexical_absolute(Path(value), base=receipt_directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="participant workspace containing data_manifest.json and data/; defaults to output/ parent",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--agent-model", default="DeepSeek-V4-Flash")
    parser.add_argument(
        "--run-receipt",
        type=Path,
        required=True,
        help="run_receipt.json binding this output to a completed execution",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = lexical_absolute(args.output)
    workspace = lexical_absolute(args.data_root) if args.data_root else output.parent
    work = lexical_absolute(args.work_dir)
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)

    task_assets = (HERE / "tasks" / args.task_id).resolve()
    anchor = task_assets / "validation_anchors.json"
    require(anchor.is_file() and not anchor.is_symlink(), f"missing evaluator asset: {args.task_id}")
    anchor_data = load_json(anchor)
    require(
        anchor_data.get("schema") == "truthinsightbench-validation-anchors"
        and anchor_data.get("release") == "V1.0"
        and anchor_data.get("task_id") == args.task_id,
        f"invalid evaluator asset identity: {anchor}",
    )

    run_receipt_path = lexical_absolute(args.run_receipt)
    run_receipt = load_json(run_receipt_path)
    require(
        run_receipt.get("schema") == "truthinsightbench-agent-run-receipt"
        and run_receipt.get("release") == "V1.0",
        "invalid Agent run receipt",
    )
    require(
        run_receipt.get("task_id") == args.task_id
        and run_receipt.get("agent_id") == args.agent,
        "Agent run receipt identity mismatch",
    )
    receipt_output = receipt_path(
        run_receipt,
        "output_path",
        receipt_directory=run_receipt_path.parent,
    )
    receipt_workspace = receipt_path(
        run_receipt,
        "workspace",
        receipt_directory=run_receipt_path.parent,
    )
    require(receipt_output == output, "Agent run receipt output path mismatch")
    require(receipt_workspace == workspace, "Agent run receipt workspace path mismatch")
    require(
        run_receipt.get("status") == "completed"
        and run_receipt.get("platform_exit_code") == 0
        and run_receipt.get("output_validation_passed") is True
        and run_receipt.get("workspace_input_integrity_passed") is True,
        "Agent run did not complete with valid output and unchanged inputs",
    )

    output_snapshot = tree_snapshot(output)
    require(
        not output_snapshot["unsafe_entries"],
        "unsafe output tree: " + ", ".join(output_snapshot["unsafe_entries"]),
    )
    require(
        run_receipt.get("output_tree_sha256") == output_snapshot["sha256"]
        and run_receipt.get("output_file_count") == output_snapshot["file_count"]
        and run_receipt.get("output_bytes") == output_snapshot["bytes"],
        "Agent run receipt output commitment mismatch",
    )

    workspace_validation = validate_workspace(workspace, expected_task_id=args.task_id)
    require(
        workspace_validation["passed"],
        "workspace input integrity failed:\n- " + "\n- ".join(workspace_validation["errors"]),
    )
    require(
        run_receipt.get("workspace_input_tree_sha256")
        == workspace_validation["input_tree_sha256"]
        and run_receipt.get("workspace_input_file_count")
        == workspace_validation["input_file_count"]
        and run_receipt.get("workspace_input_bytes") == workspace_validation["input_bytes"],
        "Agent run receipt workspace-input commitment mismatch",
    )
    require(
        workspace_validation.get("t0") == anchor_data.get("t0"),
        "workspace and evaluator T0 mismatch",
    )

    registry = load_json(REPOSITORY_ROOT / "registry.json")
    record = next(
        (row for row in registry.get("tasks", []) if row.get("task_id") == args.task_id),
        None,
    )
    require(record is not None, f"task is not registered in V1.0: {args.task_id}")
    require(
        workspace_validation["data_manifest_sha256"] == record["data_manifest_sha256"]
        and sha256_file(workspace / "ResearchObjective.md")
        == record["research_objective_sha256"]
        and workspace_validation["t0"] == record["t0"],
        "workspace is not the registered V1.0 task snapshot",
    )

    validation_path = work / "submission_validation.json"
    check = subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "submission_format" / "validate_submission.py"),
            str(output),
            "--receipt",
            str(validation_path),
        ],
        text=True,
        capture_output=True,
    )
    if check.returncode:
        print(check.stdout, end="")
        print(check.stderr, end="", file=sys.stderr)
        raise SystemExit(check.returncode)
    validation = load_json(validation_path)
    require(validation.get("passed") is True, "submission validation did not pass")
    report = validation["authoritative_report"]

    input_integrity = {
        "passed": True,
        "task_id": args.task_id,
        "t0": workspace_validation["t0"],
        "data_manifest_sha256": workspace_validation["data_manifest_sha256"],
        "workspace_input_tree_sha256": workspace_validation["input_tree_sha256"],
        "workspace_input_file_count": workspace_validation["input_file_count"],
        "workspace_input_bytes": workspace_validation["input_bytes"],
        "missing_files": [],
        "changed_files": [],
    }
    manifest = {
        "schema": "truthinsightbench-scoring-run",
        "release": "V1.0",
        "agent": args.agent,
        "task_id": args.task_id,
        "t0": anchor_data["t0"],
        "model": run_receipt.get("model", args.agent_model),
        "status": run_receipt["status"],
        "platform_exit_code": run_receipt["platform_exit_code"],
        "canonical_output_valid": True,
        "validation": validation,
        "input_integrity": input_integrity,
        "agent_run_receipt_sha256": sha256_file(run_receipt_path),
    }
    manifest_path = work / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    key = f"{args.agent}--{args.task_id}"
    run = {
        "key": key,
        "task_id": args.task_id,
        "agent": args.agent,
        "model": manifest["model"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_root": str(output),
        "report_path": report["path"],
        "report_sha256": report["sha256"],
        "finding_count": validation["registered_findings"]["count"],
        "gold_path": str(anchor),
        "gold_sha256": sha256_file(anchor),
    }
    cohort = {
        "schema": "truthinsightbench-scoring-cohort",
        "release": "V1.0",
        "agents": [args.agent],
        "summary": {
            "run_count": 1,
            "complete_task_count": 1,
            "complete_tasks": [args.task_id],
            "provisional_task_count": 0,
            "provisional_tasks": [],
        },
        "runs": [run],
    }
    cohort_path = work / "cohort.json"
    cohort_path.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "cohort": str(cohort_path),
                "scoring_output_directory": str(work / "score"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
