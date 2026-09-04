#!/usr/bin/env python3
"""Execute one Agent command against an assembled workspace and freeze its output."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

from workspace_integrity import (
    redact_secret_file,
    redact_secret_tree,
    secret_occurrences,
    sha256_file,
    tree_snapshot,
    validate_workspace,
)


AGENTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENTS_ROOT.parent


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def outside_repository(path: Path) -> bool:
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-id")
    source.add_argument("--workspace", type=Path, help="preassembled participant workspace")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--harness-version", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--command-json",
        required=True,
        help="JSON argv array supporting {workspace}, {prompt}, {output}, and {run_root}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="wall-clock limit; 0 disables the limit",
    )
    parser.add_argument(
        "--prepare-container-io",
        action="store_true",
        help="make only output/ and runtime/ writable by the unprivileged container user",
    )
    parser.add_argument(
        "--timeout-cleanup-command-json",
        help="optional JSON argv array executed only after a wall-clock timeout",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not outside_repository(run_root):
        raise SystemExit("run root must be outside the benchmark repository")
    if run_root.exists():
        raise SystemExit(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    workspace = run_root / "workspace"

    if args.workspace:
        original = args.workspace.resolve()
        original_validation = validate_workspace(original)
        if not original_validation["passed"]:
            raise SystemExit(
                "invalid preassembled workspace:\n- "
                + "\n- ".join(original_validation["errors"])
            )
        shutil.copytree(original, workspace)
        task_id = original_validation["task_id"]
        (workspace / "output").mkdir(exist_ok=True)
    else:
        task_id = args.task_id
        assembled = subprocess.run(
            [
                sys.executable,
                str(AGENTS_ROOT / "assemble_workspace.py"),
                "--task-id",
                task_id,
                "--destination",
                str(workspace),
            ],
            text=True,
            capture_output=True,
        )
        if assembled.returncode:
            raise SystemExit(assembled.stdout + assembled.stderr)

    input_before = validate_workspace(workspace, expected_task_id=task_id)
    if not input_before["passed"]:
        raise SystemExit(
            "assembled workspace failed integrity validation:\n- "
            + "\n- ".join(input_before["errors"])
        )
    if args.prepare_container_io:
        for path in sorted(workspace.rglob("*"), reverse=True):
            relative = path.relative_to(workspace)
            if relative.parts and relative.parts[0] == "output":
                continue
            path.chmod(0o555 if path.is_dir() else 0o444)
        workspace.chmod(0o555)
        (workspace / "output").chmod(0o777)
        runtime_directory = run_root / "runtime"
        runtime_directory.mkdir()
        runtime_directory.chmod(0o777)

    try:
        argv = json.loads(args.command_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --command-json: {exc}") from exc
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise SystemExit("--command-json must be a non-empty JSON string array")

    replacements = {
        "{workspace}": str(workspace),
        "{prompt}": str(workspace / "LaunchPrompt.md"),
        "{output}": str(workspace / "output"),
        "{run_root}": str(run_root),
    }
    resolved_argv = []
    for item in argv:
        for token, value in replacements.items():
            item = item.replace(token, value)
        resolved_argv.append(item)

    cleanup_argv: list[str] | None = None
    if args.timeout_cleanup_command_json:
        try:
            cleanup_argv = json.loads(args.timeout_cleanup_command_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --timeout-cleanup-command-json: {exc}") from exc
        if (
            not isinstance(cleanup_argv, list)
            or not cleanup_argv
            or not all(isinstance(item, str) and item for item in cleanup_argv)
        ):
            raise SystemExit("--timeout-cleanup-command-json must be a non-empty JSON string array")

    started = now()
    timed_out = False
    timeout_cleanup_exit_code: int | None = None
    exit_code: int | None = None
    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            resolved_argv,
            cwd=workspace,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name == "posix",
        )
        try:
            exit_code = process.wait(
                timeout=None if args.timeout_seconds == 0 else args.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            if cleanup_argv:
                try:
                    cleanup = subprocess.run(
                        cleanup_argv,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                    timeout_cleanup_exit_code = cleanup.returncode
                except subprocess.TimeoutExpired:
                    timeout_cleanup_exit_code = -1

    configured_secrets = [
        os.environ.get(name, "")
        for name in (
            "TIB_MODEL_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "OPENAI_API_KEY",
        )
    ]
    redacted_runtime_files = redact_secret_tree(run_root / "runtime", configured_secrets)
    redacted_log_files = [
        path.name
        for path in (stdout_path, stderr_path)
        if redact_secret_file(path, configured_secrets)
    ]
    output_secret_paths = secret_occurrences(workspace / "output", configured_secrets)

    validation_path = run_root / "submission_validation.json"
    validation = subprocess.run(
        [
            sys.executable,
            str(AGENTS_ROOT / "validate_output.py"),
            str(workspace / "output"),
            "--receipt",
            str(validation_path),
        ],
        text=True,
        capture_output=True,
    )
    valid = validation.returncode == 0
    output_snapshot = tree_snapshot(workspace / "output")
    input_after = validate_workspace(workspace, expected_task_id=task_id)
    input_unchanged = (
        input_after["passed"]
        and input_after["input_tree_sha256"] == input_before["input_tree_sha256"]
    )
    status = (
        "timed_out"
        if timed_out
        else "completed"
        if exit_code == 0
        and valid
        and input_unchanged
        and not output_snapshot["unsafe_entries"]
        and not output_secret_paths
        else "failed"
    )
    receipt = {
        "schema": "truthinsightbench-agent-run-receipt",
        "release": "V1.0",
        "agent_id": args.agent_id,
        "harness": args.harness,
        "harness_version": args.harness_version,
        "task_id": task_id,
        "model": args.model,
        "thinking_enabled": False,
        "status": status,
        "started_at": started,
        "finished_at": now(),
        "timeout_seconds": args.timeout_seconds,
        "platform_exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_cleanup_exit_code": timeout_cleanup_exit_code,
        "credential_redaction": {
            "runtime_file_count": len(redacted_runtime_files),
            "log_files": redacted_log_files,
            "output_secret_paths": output_secret_paths,
        },
        "output_validation_passed": valid,
        "workspace_input_integrity_passed": input_unchanged,
        "workspace_input_tree_sha256": input_before["input_tree_sha256"],
        "workspace_input_file_count": input_before["input_file_count"],
        "workspace_input_bytes": input_before["input_bytes"],
        "workspace_input_integrity_errors": input_after["errors"],
        "workspace": "workspace",
        "output_path": "workspace/output",
        "output_tree_sha256": output_snapshot["sha256"],
        "output_file_count": output_snapshot["file_count"],
        "output_bytes": output_snapshot["bytes"],
        "output_tree_integrity_errors": output_snapshot["unsafe_entries"],
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "command": {"executable": resolved_argv[0], "argument_count": len(resolved_argv) - 1},
        "submission_validation_receipt": "submission_validation.json",
    }
    (run_root / "run_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
