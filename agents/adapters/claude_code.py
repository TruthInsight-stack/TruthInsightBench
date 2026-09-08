#!/usr/bin/env python3
"""Claude Code 2.1.220 adapter for an assembled participant workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common import executable, model_endpoint, redact_secret_tree, require_version


EXPECTED_VERSION = "2.1.220"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()

    binary = executable("TIB_CLAUDE_BIN", "claude")
    require_version([binary, "--version"], EXPECTED_VERSION, "Claude Code")
    workspace = args.workspace.resolve()
    runtime = args.runtime_root.resolve() / "claude"
    runtime.mkdir(parents=True, exist_ok=True)
    base_url, api_key = model_endpoint()
    env = {
        **os.environ,
        "HOME": str(runtime),
        "ANTHROPIC_BASE_URL": base_url.removesuffix("/v1"),
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    prompt = args.prompt_file.read_text(encoding="utf-8")
    command = [
        binary,
        "--print",
        "--bare",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--model",
        args.model,
        "--tools",
        "default",
        "--output-format",
        "stream-json",
        "--verbose",
        prompt,
    ]
    try:
        return subprocess.run(command, cwd=workspace, env=env).returncode
    finally:
        redact_secret_tree(runtime, [api_key])


if __name__ == "__main__":
    raise SystemExit(main())
