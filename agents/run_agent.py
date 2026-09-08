#!/usr/bin/env python3
"""Run one of the four published Agent harness profiles on one task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


AGENTS_ROOT = Path(__file__).resolve().parent


def main() -> int:
    profiles = json.loads((AGENTS_ROOT / "agent_profiles.json").read_text(encoding="utf-8"))
    available = profiles["agents"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(available), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model", default=profiles["default_participant_model"])
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument(
        "--container-image",
        help="run the adapter in this Docker image with only the workspace and public Agent code mounted",
    )
    args = parser.parse_args()

    profile = available[args.agent]
    adapter = (AGENTS_ROOT / profile["adapter"]).resolve()
    adapter_arguments = [
        "--workspace",
        "{workspace}",
        "--prompt-file",
        "{prompt}",
        "--model",
        args.model,
        "--runtime-root",
        "{run_root}/runtime",
    ]
    if args.container_image:
        name_fragment = re.sub(r"[^a-z0-9_.-]+", "-", f"{args.agent}-{args.task_id}".lower())
        run_hash = hashlib.sha256(str(args.run_root.resolve()).encode()).hexdigest()[:12]
        container_name = f"tib-{name_fragment[:80]}-{run_hash}"
        host_uid = os.getuid() if hasattr(os, "getuid") else 1000
        host_gid = os.getgid() if hasattr(os, "getgid") else 1000
        container_uid = 1000 if host_uid == 0 else host_uid
        container_gid = 1000 if host_gid == 0 else host_gid
        container_adapter = f"/opt/truthinsightbench/agents/{profile['adapter']}"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--user",
            f"{container_uid}:{container_gid}",
            "--network",
            "host",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-e",
            "TIB_MODEL_BASE_URL",
            "-e",
            "TIB_MODEL_API_KEY",
            "-v",
            "{workspace}:/workspace",
            "-v",
            f"{AGENTS_ROOT}:/opt/truthinsightbench/agents:ro",
            "-v",
            "{run_root}/runtime:/runtime",
            "-w",
            "/workspace",
            args.container_image,
            "python3",
            container_adapter,
            "--workspace",
            "/workspace",
            "--prompt-file",
            "/workspace/LaunchPrompt.md",
            "--model",
            args.model,
            "--runtime-root",
            "/runtime",
        ]
    else:
        command = [sys.executable, str(adapter), *adapter_arguments]
    timeout = profile["default_timeout_seconds"] if args.timeout_seconds is None else args.timeout_seconds
    invocation = [
        sys.executable,
        str(AGENTS_ROOT / "run_command.py"),
        "--task-id",
        args.task_id,
        "--agent-id",
        args.agent,
        "--harness",
        profile["harness"],
        "--harness-version",
        profile["version"],
        "--model",
        args.model,
        "--run-root",
        str(args.run_root),
        "--timeout-seconds",
        str(timeout),
        "--command-json",
        json.dumps(command),
    ]
    if args.container_image:
        invocation.extend(
            [
                "--prepare-container-io",
                "--timeout-cleanup-command-json",
                json.dumps(["docker", "rm", "--force", container_name]),
            ]
        )
    return subprocess.run(invocation).returncode


if __name__ == "__main__":
    raise SystemExit(main())
