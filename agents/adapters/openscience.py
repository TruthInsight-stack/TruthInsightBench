#!/usr/bin/env python3
"""OpenScience 2.0.1 adapter with bounded delivery-only continuation turns."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common import (
    continuation_prompt,
    executable,
    model_endpoint,
    output_validation,
    redact_secret_tree,
    require_version,
    write_json,
)


EXPECTED_VERSION = "2.0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--max-continuations", type=int, default=8)
    args = parser.parse_args()

    binary = executable("TIB_OPENSCIENCE_BIN", "openscience")
    require_version([binary, "--version"], EXPECTED_VERSION, "OpenScience")
    workspace = args.workspace.resolve()
    runtime = args.runtime_root.resolve() / "openscience"
    xdg = {name: runtime / name for name in ("config", "data", "state", "cache")}
    for path in xdg.values():
        path.mkdir(parents=True, exist_ok=True)
    base_url, api_key = model_endpoint()
    env = {
        **os.environ,
        "HOME": str(runtime),
        "XDG_CONFIG_HOME": str(xdg["config"]),
        "XDG_DATA_HOME": str(xdg["data"]),
        "XDG_STATE_HOME": str(xdg["state"]),
        "XDG_CACHE_HOME": str(xdg["cache"]),
    }
    try:
        configured = subprocess.run(
            [
                binary,
                "local",
                "add",
                "--url",
                f"{base_url.removesuffix('/v1')}/v1",
                "--model",
                args.model,
                "--id",
                "truthinsightbench",
                "--key",
                api_key,
                "--default",
            ],
            cwd=workspace,
            env=env,
        )
        if configured.returncode:
            return configured.returncode

        turns = []
        prompt = args.prompt_file.read_text(encoding="utf-8")
        for index in range(args.max_continuations + 1):
            command = [
                binary,
                "run",
                "--model",
                f"truthinsightbench/{args.model}",
                "--agent",
                "research",
                "--format",
                "json",
            ]
            if index:
                command.append("--continue")
            command.append(prompt)
            completed = subprocess.run(command, cwd=workspace, env=env)
            validation = output_validation(
                workspace, runtime / f"validation-{index + 1:02d}.json"
            )
            turns.append(
                {
                    "turn": index + 1,
                    "platform_exit_code": completed.returncode,
                    "output_validation_passed": validation.get("passed") is True,
                    "validation_errors": validation.get("errors", []),
                }
            )
            write_json(
                runtime / "adapter_receipt.json",
                {
                    "schema": "truthinsightbench-openscience-adapter-receipt",
                    "release": "V1.0",
                    "max_continuations": args.max_continuations,
                    "turns": turns,
                },
            )
            if completed.returncode:
                return completed.returncode
            if validation.get("passed") is True:
                return 0
            prompt = continuation_prompt(validation.get("errors", []))
        return 2
    finally:
        redact_secret_tree(runtime, [api_key])


if __name__ == "__main__":
    raise SystemExit(main())
