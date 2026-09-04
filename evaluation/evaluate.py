#!/usr/bin/env python3
"""Validate, prepare, and score one or more frozen TruthInsightBench runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parent
EVALUATOR_ROOT = EVALUATION_ROOT / "evaluator"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, env=env)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def require_node_20() -> None:
    binary = shutil.which("node")
    if not binary:
        raise SystemExit("Node.js 20 or newer is required")
    completed = subprocess.run([binary, "--version"], text=True, capture_output=True)
    try:
        major = int(completed.stdout.strip().lstrip("v").split(".", 1)[0])
    except (ValueError, IndexError):
        raise SystemExit("unable to determine the installed Node.js version") from None
    if completed.returncode or major < 20:
        raise SystemExit(
            f"Node.js 20 or newer is required; detected {completed.stdout.strip() or 'unknown'}"
        )


def output_from_receipt(run_root: Path, receipt: dict) -> Path:
    declared = Path(receipt.get("output_path", ""))
    candidate = declared if declared.is_absolute() else run_root / declared
    return Path(os.path.abspath(candidate))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--judge-endpoint", default=os.environ.get("JUDGE_ENDPOINT"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "Apsara-Stack/GLM-5.1-W4A8"))
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--closure-concurrency", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="validate every input without calling a model")
    args = parser.parse_args()

    require_node_20()
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)

    cohorts: list[Path] = []
    identities: list[dict] = []
    for run_root_arg in args.run_root:
        run_root = run_root_arg.resolve()
        receipt_path = run_root / "run_receipt.json"
        if not receipt_path.is_file():
            raise SystemExit(f"missing run receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "truthinsightbench-agent-run-receipt":
            raise SystemExit(f"invalid run receipt: {receipt_path}")
        identity = {"agent": receipt["agent_id"], "task_id": receipt["task_id"]}
        identities.append(identity)
        prepared = work / "prepared" / f"{identity['agent']}--{identity['task_id']}"
        command = [
            sys.executable,
            str(EVALUATOR_ROOT / "prepare_submission.py"),
            "--task-id",
            identity["task_id"],
            "--agent",
            identity["agent"],
            "--agent-model",
            receipt.get("model", "unspecified"),
            "--output",
            str(output_from_receipt(run_root, receipt)),
            "--data-root",
            str(run_root / receipt.get("workspace", "workspace")),
            "--run-receipt",
            str(receipt_path),
            "--work-dir",
            str(prepared),
        ]
        run(command)
        cohorts.append(prepared / "cohort.json")

    cohort_path = work / "cohort.json"
    merge = [sys.executable, str(EVALUATOR_ROOT / "merge_cohorts.py")]
    for cohort in cohorts:
        merge.extend(["--cohort", str(cohort)])
    merge.extend(["--output", str(cohort_path)])
    run(merge)

    environment = {
        **os.environ,
        "JUDGE_MODEL": args.judge_model,
        "JUDGE_CONCURRENCY": str(args.judge_concurrency),
        "CLOSURE_CONCURRENCY": str(args.closure_concurrency),
        "SCORING_COHORT_PATH": str(cohort_path),
        "SCORING_OUT_DIR": str(work / "score"),
    }
    if args.judge_endpoint:
        environment["JUDGE_ENDPOINT"] = args.judge_endpoint
    if args.dry_run:
        environment["DRY_RUN"] = "1"
        run(["node", str(EVALUATOR_ROOT / "score_cohort.mjs")], env=environment)
        print(
            json.dumps(
                {
                    "status": "READY",
                    "mode": "dry_run",
                    "run_count": len(identities),
                    "cohort": str(cohort_path),
                    "external_model_called": False,
                },
                indent=2,
            )
        )
        return 0
    if not args.judge_endpoint:
        raise SystemExit("--judge-endpoint or JUDGE_ENDPOINT is required for scoring")

    run(["node", str(EVALUATOR_ROOT / "score_cohort.mjs")], env=environment)
    final_environment = {
        **environment,
        "CLOSURE_BASE_SCORE_PATH": str(work / "score" / "scoring_result.json"),
        "CLOSURE_COHORT_PATH": str(cohort_path),
        "CLOSURE_OUT_DIR": str(work / "final"),
    }
    run(["node", str(EVALUATOR_ROOT / "finalize_scores.mjs")], env=final_environment)
    final_path = work / "final" / "final_scoring_result.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    scores = [
        {
            "agent": row["agent"],
            "task_id": row["task_id"],
            "score": row["run"]["run_score_0_100"],
        }
        for row in final["run_results"]
    ]
    print(
        json.dumps(
            {
                "status": "SCORED",
                "judge_model": args.judge_model,
                "run_count": len(scores),
                "scores": scores,
                "result": str(final_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
