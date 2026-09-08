"""Shared helpers for the published Agent adapters."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


AGENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_ROOT))
from workspace_integrity import redact_secret_tree  # noqa: E402


TRUSTED_VALIDATOR = Path(__file__).resolve().parents[1] / "validate_output.py"


def executable(env_name: str, default: str) -> str:
    candidate = os.environ.get(env_name, default)
    resolved = shutil.which(candidate) if not Path(candidate).is_file() else candidate
    if not resolved:
        raise SystemExit(f"required executable not found: {candidate} (override with {env_name})")
    return str(Path(resolved).resolve())


def require_version(command: list[str], expected: str, label: str) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    rendered = (completed.stdout or completed.stderr).strip().splitlines()
    actual = rendered[0] if rendered else ""
    if completed.returncode or expected not in actual:
        raise SystemExit(f"{label} {expected} is required; detected: {actual or 'unavailable'}")


def model_endpoint() -> tuple[str, str]:
    base_url = os.environ.get("TIB_MODEL_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("TIB_MODEL_API_KEY", "").strip()
    if not base_url:
        raise SystemExit("TIB_MODEL_BASE_URL is required")
    if not api_key:
        raise SystemExit("TIB_MODEL_API_KEY is required")
    return base_url, api_key


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def output_validation(workspace: Path, receipt: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(TRUSTED_VALIDATOR),
            str(workspace / "output"),
            "--receipt",
            str(receipt),
        ],
        text=True,
        capture_output=True,
    )
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    return {"passed": False, "errors": [completed.stderr.strip() or "validator_failed"]}


def continuation_prompt(errors: list[str]) -> str:
    rendered = "\n".join(f"- {item}" for item in errors[:16])
    return f"""Continue the same benchmark task from the files already present in this workspace. The preceding turn ended before the public delivery contract passed. Do not restart the scientific analysis or change a data-warranted conclusion merely to satisfy structure.

Current delivery validation errors:
{rendered}

Before ending, make the root Markdown report, 2 to 5 clearly labelled findings, executed analysis source, inspectable result evidence, a reproduction trace, and per-finding evidence resolution satisfy `python3 validate_output.py output`. Preserve the Agent's native layout."""
