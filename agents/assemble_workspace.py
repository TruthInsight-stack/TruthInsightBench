#!/usr/bin/env python3
"""Assemble one participant-visible TruthInsightBench workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


AGENTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = AGENTS_ROOT.parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")

    registry = load_json(REPOSITORY_ROOT / "registry.json")
    record = next(
        (row for row in registry["tasks"] if row["task_id"] == args.task_id),
        None,
    )
    if record is None:
        raise SystemExit(f"unknown task: {args.task_id}")

    task_root = REPOSITORY_ROOT / "tasks" / args.task_id
    manifest_path = task_root / "data_manifest.json"
    objective_path = task_root / "ResearchObjective.md"
    guide_path = task_root / "DATA_GUIDE.md"
    manifest = load_json(manifest_path)

    if manifest.get("task_id") != args.task_id or manifest.get("t0") != record["t0"]:
        raise SystemExit(f"task identity mismatch: {args.task_id}")
    if sha256(manifest_path) != record["data_manifest_sha256"]:
        raise SystemExit(f"manifest checksum mismatch: {args.task_id}")
    if sha256(objective_path) != record["research_objective_sha256"]:
        raise SystemExit(f"research objective checksum mismatch: {args.task_id}")

    for item in manifest["files"]:
        source = task_root / "data" / item["path"]
        if not source.is_file():
            raise SystemExit(f"missing scientific file: {item['path']}")
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise SystemExit(f"scientific payload mismatch: {item['path']}")

    destination.mkdir(parents=True)
    copy(AGENTS_ROOT / "PublicTaskContract.md", destination / "PublicTaskContract.md")
    copy(objective_path, destination / "ResearchObjective.md")
    copy(guide_path, destination / "DATA_GUIDE.md")
    copy(manifest_path, destination / "data_manifest.json")
    copy(AGENTS_ROOT / "validate_output.py", destination / "validate_output.py")
    copy(AGENTS_ROOT / "launcher" / "effective_user_prompt.txt", destination / "LaunchPrompt.md")
    for item in manifest["files"]:
        source = task_root / "data" / item["path"]
        target = destination / "data" / item["path"]
        copy(source, target)
        if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            raise SystemExit(f"copied scientific payload mismatch: {item['path']}")
    (destination / "output").mkdir()

    receipt = {
        "schema": "truthinsightbench-participant-workspace",
        "release": "V1.0",
        "task_id": args.task_id,
        "t0": record["t0"],
        "scientific_file_count": len(manifest["files"]),
        "scientific_bytes": sum(item["bytes"] for item in manifest["files"]),
        "data_manifest_sha256": sha256(destination / "data_manifest.json"),
        "research_objective_sha256": sha256(destination / "ResearchObjective.md"),
        "launch_prompt_sha256": sha256(destination / "LaunchPrompt.md"),
    }
    (destination / "workspace_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "READY", "workspace": str(destination), **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
