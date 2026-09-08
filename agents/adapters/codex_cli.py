#!/usr/bin/env python3
"""Codex CLI 0.149.0 adapter for an assembled participant workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common import executable, model_endpoint, redact_secret_tree, require_version


EXPECTED_VERSION = "0.149.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()

    binary = executable("TIB_CODEX_BIN", "codex")
    require_version([binary, "--version"], EXPECTED_VERSION, "Codex CLI")
    workspace = args.workspace.resolve()
    runtime = args.runtime_root.resolve() / "codex"
    runtime.mkdir(parents=True, exist_ok=True)
    base_url, api_key = model_endpoint()
    (runtime / "config.toml").write_text(
        "\n".join(
            [
                f'model = "{args.model}"',
                'model_provider = "truthinsightbench"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                'check_for_update_on_startup = false',
                '',
                '[model_providers.truthinsightbench]',
                'name = "TruthInsightBench participant endpoint"',
                f'base_url = "{base_url.removesuffix("/v1")}/v1"',
                'env_key = "TIB_MODEL_API_KEY"',
                'wire_api = "responses"',
                'request_max_retries = 4',
                'stream_max_retries = 4',
                'stream_idle_timeout_ms = 300000',
                '',
            ]
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "CODEX_HOME": str(runtime), "TIB_MODEL_API_KEY": api_key, "NO_COLOR": "1"}
    prompt = args.prompt_file.read_text(encoding="utf-8")
    command = [
        binary,
        "exec",
        "--skip-git-repo-check",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
        "--ephemeral",
        "--json",
        "-m",
        args.model,
        "-C",
        str(workspace),
        "-",
    ]
    try:
        return subprocess.run(command, cwd=workspace, env=env, input=prompt, text=True).returncode
    finally:
        redact_secret_tree(runtime, [api_key])


if __name__ == "__main__":
    raise SystemExit(main())
