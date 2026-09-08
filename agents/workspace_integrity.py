"""Integrity helpers shared by the Agent runner and evaluator.

The functions in this module never execute files from an Agent workspace.
They accept regular files and directories only, bind every scientific payload
to ``data_manifest.json``, and produce deterministic tree commitments.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_INPUTS = (
    "PublicTaskContract.md",
    "ResearchObjective.md",
    "DATA_GUIDE.md",
    "data_manifest.json",
    "validate_output.py",
    "LaunchPrompt.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_secrets(secrets: list[str]) -> list[bytes]:
    return sorted({value.encode() for value in secrets if len(value) >= 8}, key=len, reverse=True)


def redact_secret_file(path: Path, secrets: list[str]) -> bool:
    """Replace exact configured credentials in a regular file."""

    encoded = normalized_secrets(secrets)
    if not encoded or path.is_symlink() or not path.is_file():
        return False
    try:
        content = path.read_bytes()
    except OSError:
        return False
    redacted = content
    for secret in encoded:
        redacted = redacted.replace(secret, b"[REDACTED_BY_TRUTHINSIGHTBENCH]")
    if redacted == content:
        return False
    path.write_bytes(redacted)
    return True


def redact_secret_tree(root: Path, secrets: list[str]) -> list[str]:
    """Redact exact configured credentials from regular files below *root*."""

    if not root.is_dir() or root.is_symlink():
        return []
    redacted: list[str] = []
    for path in root.rglob("*"):
        if redact_secret_file(path, secrets):
            redacted.append(path.relative_to(root).as_posix())
    return sorted(redacted)


def secret_occurrences(root: Path, secrets: list[str]) -> list[str]:
    """Return output paths containing an exact configured credential."""

    encoded = normalized_secrets(secrets)
    if not encoded or not root.is_dir() or root.is_symlink():
        return []
    matches: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if any(secret in content for secret in encoded):
            matches.append(path.relative_to(root).as_posix())
    return sorted(matches)


def tree_snapshot(root: Path, *, exclude_top_level: set[str] | None = None) -> dict[str, Any]:
    """Commit to regular files below *root* without following symbolic links."""

    excluded = exclude_top_level or set()
    digest = hashlib.sha256()
    count = 0
    size = 0
    unsafe_entries: list[str] = []
    if root.is_symlink():
        return {
            "sha256": digest.hexdigest(),
            "file_count": 0,
            "bytes": 0,
            "unsafe_entries": ["symbolic_link:."],
        }
    if not root.is_dir():
        return {
            "sha256": digest.hexdigest(),
            "file_count": 0,
            "bytes": 0,
            "unsafe_entries": ["missing_directory"],
        }

    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if relative.parts and relative.parts[0] in excluded:
            continue
        rendered = relative.as_posix()
        if candidate.is_symlink():
            unsafe_entries.append(f"symbolic_link:{rendered}")
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            unsafe_entries.append(f"non_regular_file:{rendered}")
            continue
        file_size = candidate.stat().st_size
        digest.update(f"{rendered}\0{file_size}\0{sha256_file(candidate)}\n".encode())
        count += 1
        size += file_size
    return {
        "sha256": digest.hexdigest(),
        "file_count": count,
        "bytes": size,
        "unsafe_entries": unsafe_entries,
    }


def _safe_manifest_path(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return None
    relative = Path(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative.as_posix()


def validate_workspace(workspace: Path, *, expected_task_id: str | None = None) -> dict[str, Any]:
    """Validate a participant workspace and return its frozen input snapshot."""

    workspace = workspace.resolve()
    errors: list[str] = []
    if not workspace.is_dir():
        errors.append("missing_workspace")
        return {"passed": False, "errors": errors}

    for name in REQUIRED_INPUTS:
        path = workspace / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing_or_unsafe_required_input:{name}")
    data_root = workspace / "data"
    if data_root.is_symlink() or not data_root.is_dir():
        errors.append("missing_or_unsafe_data_directory")

    manifest: dict[str, Any] = {}
    manifest_path = workspace / "data_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("root must be an object")
            manifest = loaded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid_data_manifest:{exc}")

    task_id = manifest.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        errors.append("missing_manifest_task_id")
    elif expected_task_id is not None and task_id != expected_task_id:
        errors.append(f"task_id_mismatch:expected={expected_task_id}:actual={task_id}")

    rows = manifest.get("files")
    declared: set[str] = set()
    if not isinstance(rows, list) or not rows:
        errors.append("manifest_files_must_be_nonempty_array")
        rows = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            errors.append(f"manifest_file_not_object:{index}")
            continue
        relative = _safe_manifest_path(item.get("path"))
        if relative is None:
            errors.append(f"unsafe_manifest_path:{index}")
            continue
        if relative in declared:
            errors.append(f"duplicate_manifest_path:{relative}")
            continue
        declared.add(relative)
        target = data_root / relative
        if target.is_symlink() or not target.is_file():
            errors.append(f"missing_or_unsafe_scientific_file:{relative}")
            continue
        expected_bytes = item.get("bytes")
        expected_sha256 = item.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            errors.append(f"invalid_manifest_bytes:{relative}")
        elif target.stat().st_size != expected_bytes:
            errors.append(f"scientific_file_size_mismatch:{relative}")
        if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(expected_sha256):
            errors.append(f"invalid_manifest_sha256:{relative}")
        elif sha256_file(target) != expected_sha256:
            errors.append(f"scientific_file_sha256_mismatch:{relative}")

    actual: set[str] = set()
    if data_root.is_dir() and not data_root.is_symlink():
        for candidate in sorted(data_root.rglob("*")):
            relative = candidate.relative_to(data_root).as_posix()
            if candidate.is_symlink():
                errors.append(f"unsafe_data_symbolic_link:{relative}")
            elif candidate.is_file():
                actual.add(relative)
            elif not candidate.is_dir():
                errors.append(f"unsafe_data_non_regular_file:{relative}")
    for relative in sorted(declared - actual):
        errors.append(f"manifest_file_not_present:{relative}")
    for relative in sorted(actual - declared):
        errors.append(f"unmanifested_scientific_file:{relative}")

    snapshot = tree_snapshot(workspace, exclude_top_level={"output"})
    errors.extend(snapshot["unsafe_entries"])
    return {
        "passed": not errors,
        "errors": errors,
        "task_id": task_id,
        "t0": manifest.get("t0"),
        "data_manifest_sha256": sha256_file(manifest_path)
        if manifest_path.is_file() and not manifest_path.is_symlink()
        else None,
        "input_tree_sha256": snapshot["sha256"],
        "input_file_count": snapshot["file_count"],
        "input_bytes": snapshot["bytes"],
    }
