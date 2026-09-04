#!/usr/bin/env python3
"""DeepSeek Harness 0.1.0rc7 adapter with delivery-only continuation turns."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
AGENTS_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(AGENTS_ROOT))
from common import redact_secret_tree  # noqa: E402
from validate_output import validate_output  # noqa: E402


EXPECTED_VERSION = "0.1.0rc7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-continuations", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def continuation_prompt(errors: list[str]) -> str:
    rendered = "\n".join(f"- {item}" for item in errors[:16])
    return f"""Continue the same benchmark task from the files already present in this workspace. The preceding DeepSeek Harness turn ended before the public semantic delivery gate passed. Do not restart the scientific analysis, change any frozen input, or change data-warranted conclusions merely to satisfy structure.

Current public semantic delivery validation errors:
{rendered}

Before ending this turn, make the root Markdown report, 2 to 5 clearly labelled findings, real analysis sources, inspectable result evidence, a reproduction trace, and per-finding evidence resolution satisfy `python3 validate_output.py output`. Preserve the Agent's native layout. This feedback contains delivery checks only; retain only scientific conclusions already supported by the workspace data."""


def main() -> int:
    args = parse_args()
    try:
        actual_version = importlib.metadata.version("deepseek-harness-sdk")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(
            f"deepseek-harness-sdk {EXPECTED_VERSION} is required; package is not installed"
        ) from None
    if actual_version != EXPECTED_VERSION:
        raise SystemExit(
            f"deepseek-harness-sdk {EXPECTED_VERSION} is required; detected: {actual_version}"
        )
    from deepseek_harness import DeepSeekHarness
    workspace = args.workspace.resolve()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    runtime = args.runtime_root.resolve() / "deepseek_harness"
    session_root = runtime / "sessions"
    session_root.mkdir(parents=True, exist_ok=True)
    response_file = runtime / "response.json"
    receipt_file = runtime / "adapter_receipt.json"
    dsh_home = runtime / "home"
    dsh_home.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("TIB_MODEL_BASE_URL")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("TIB_MODEL_API_KEY")
    if not base_url:
        raise SystemExit("TIB_MODEL_BASE_URL or DEEPSEEK_BASE_URL is required")
    if not api_key:
        raise SystemExit("TIB_MODEL_API_KEY or DEEPSEEK_API_KEY is required")
    turns: list[dict[str, Any]] = []
    final_response = ""
    task_id = read_json(workspace / "data_manifest.json")["task_id"]
    session_id = f"truthinsightbench-{task_id}"
    try:
        with DeepSeekHarness(
            provider="deepseek-official",
            model=args.model,
            max_tokens=args.max_tokens,
            cwd=str(workspace),
            runtime_cwd=str(workspace),
            session_root=str(session_root),
            cordis=str(HERE / "agent.cordis.yml"),
            base_url=base_url,
            api_key=api_key,
            env={
                "HOME": str(dsh_home),
                "XDG_CONFIG_HOME": str(dsh_home / "config"),
                "XDG_DATA_HOME": str(dsh_home / "data"),
                "XDG_STATE_HOME": str(dsh_home / "state"),
                "XDG_CACHE_HOME": str(dsh_home / "cache"),
            },
            # Bound runtime/config initialization so a broken Cordis composition
            # produces diagnostics instead of waiting forever. This value is
            # cleared immediately after startup; scientific turns have no deadline.
            request_timeout_seconds=60,
        ) as harness:
            harness.client.config.request_timeout_seconds = None
            session = harness.start_session(session_id)
            turn_prompt = prompt
            for turn_index in range(1, args.max_continuations + 2):
                result = session.run(turn_prompt)
                final_response = result.final_response
                validation = validate_output(workspace / "output")
                turns.append(
                    {
                        "turn": turn_index,
                        "finish_reason": result.finish_reason,
                        "canonical_output_valid": validation["passed"],
                        "validation_errors": validation["errors"],
                    }
                )
                write_json(
                    receipt_file,
                    {
                        "schema": "truthinsightbench-deepseek-harness-semantic-output-adapter-receipt",
                        "session_id": session_id,
                        "max_continuations": args.max_continuations,
                        "turns": turns,
                    },
                )
                write_json(
                    response_file,
                    {
                        "session_id": session_id,
                        "final_response": final_response,
                    },
                )
                if validation["passed"]:
                    break
                if result.finish_reason == "error":
                    break
                if turn_index > args.max_continuations:
                    break
                turn_prompt = continuation_prompt(validation["errors"])
    finally:
        redact_secret_tree(runtime, [api_key])

    print(final_response)
    return 0 if turns and turns[-1]["canonical_output_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
