#!/usr/bin/env python3
"""Replay all deterministic evaluator actions against the bundled task data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path, *, excluded: set[str] | None = None) -> tuple[str, int]:
    excluded = excluded or set()
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"verification tree contains symlink: {path}")
        if path.is_file():
            files.append((relative, path))
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def update_manifests() -> None:
    registry = load_json(REPOSITORY_ROOT / "registry.json")
    shared_hash, shared_count = tree_sha256(HERE / "verification_runtime")
    for record in registry["tasks"]:
        manifest_path = HERE / "tasks" / record["task_id"] / "verification" / "action_manifest.json"
        manifest = load_json(manifest_path)
        verification_hash, verification_count = tree_sha256(
            manifest_path.parent,
            excluded={"action_manifest.json"},
        )
        manifest["verification_tree_sha256"] = verification_hash
        manifest["verification_file_count"] = verification_count
        manifest["shared_runtime_tree_sha256"] = shared_hash
        manifest["shared_runtime_file_count"] = shared_count
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def update_novelty_hashes() -> None:
    registry = load_json(REPOSITORY_ROOT / "registry.json")
    pattern = re.compile(r'("query_spec_sha256"\s*:\s*")[0-9a-f]{64}(")')
    for record in registry["tasks"]:
        novelty_root = HERE / "tasks" / record["task_id"] / "novelty"
        query_spec_path = novelty_root / "novelty_query_spec.json"
        asset_path = novelty_root / "frozen_novelty_search_evidence.json"
        replacement = rf"\g<1>{sha256_file(query_spec_path)}\g<2>"
        updated, count = pattern.subn(replacement, asset_path.read_text(encoding="utf-8"))
        if count != 1:
            raise ValueError(f"expected one query_spec_sha256 field: {asset_path}")
        asset_path.write_text(updated, encoding="utf-8")


def value_at(root: Any, path: str) -> Any:
    value = root
    for token in path.split("."):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def bindings_from(anchors: dict[str, Any]) -> list[tuple[dict, dict, str]]:
    bindings: list[tuple[dict, dict, str]] = []
    for card in anchors.get("cards", []):
        bindings.append((card, card["recompute_binding"], "recompute"))
        bindings.extend(
            (card, binding, "perturb")
            for binding in card.get("perturbation_bindings", [])
        )
    return bindings


def hit_strictly_before(hit: dict[str, Any], t0: str) -> bool:
    date = str(hit.get("publication_date") or hit.get("date") or "").strip()
    if re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", date):
        return date < t0
    try:
        year = int(hit.get("publication_year") or hit.get("year"))
    except (TypeError, ValueError):
        return False
    return year < int(t0[:4])


def main() -> int:
    errors: list[str] = []
    action_runs = 0
    quantity_checks = 0
    novelty_cards = 0
    novelty_pre_t0_hits = 0
    novelty_excluded_hits = 0
    registry = load_json(REPOSITORY_ROOT / "registry.json")
    shared_runtime_hash, shared_runtime_count = tree_sha256(HERE / "verification_runtime")

    for record in registry["tasks"]:
        task_id = record["task_id"]
        task_assets = HERE / "tasks" / task_id
        participant_task = REPOSITORY_ROOT / "tasks" / task_id
        try:
            manifest = load_json(task_assets / "verification" / "action_manifest.json")
            anchors = load_json(task_assets / "validation_anchors.json")
            novelty_spec = load_json(task_assets / "novelty" / "novelty_query_spec.json")
            novelty_asset = load_json(
                task_assets / "novelty" / "frozen_novelty_search_evidence.json"
            )
        except Exception as exc:
            errors.append(f"{task_id}: asset load: {exc}")
            continue

        t0 = str(anchors.get("t0", ""))
        card_ids = [str(card.get("card_id")) for card in anchors.get("cards", [])]
        asset_cards = novelty_asset.get("cards", {})
        spec_cards = novelty_spec.get("cards", {})
        if (
            novelty_asset.get("schema")
            != "truthinsightbench-frozen-novelty-search-evidence"
            or novelty_asset.get("task_id") != task_id
            or novelty_asset.get("strict_before") != t0
            or novelty_asset.get("all_queries_completed") is not True
            or novelty_asset.get("query_spec_sha256")
            != sha256_file(task_assets / "novelty" / "novelty_query_spec.json")
            or sorted(asset_cards) != sorted(card_ids)
            or sorted(spec_cards) != sorted(card_ids)
        ):
            errors.append(f"{task_id}: novelty asset identity/coverage")
        else:
            for card_id in card_ids:
                novelty_cards += 1
                expected_queries = spec_cards[card_id].get("queries", [])
                executed_queries = asset_cards[card_id].get("queries", [])
                if [row.get("query") for row in executed_queries] != expected_queries:
                    errors.append(f"{task_id}:{card_id}: novelty query mismatch")
                for query in executed_queries:
                    if (
                        query.get("strict_before") != t0
                        or not (
                            query.get("ok") is True
                            or query.get("request_completed") is True
                        )
                    ):
                        errors.append(f"{task_id}:{card_id}: novelty query receipt")
                    for hit in query.get("hits", []):
                        if hit_strictly_before(hit, t0):
                            novelty_pre_t0_hits += 1
                        else:
                            novelty_excluded_hits += 1

        if (
            manifest.get("schema") != "truthinsightbench-verification-manifest"
            or manifest.get("release") != "V1.0"
            or manifest.get("task_id") != task_id
        ):
            errors.append(f"{task_id}: action manifest identity")

        try:
            verification_hash, verification_count = tree_sha256(
                task_assets / "verification",
                excluded={"action_manifest.json"},
            )
            if (
                manifest.get("verification_tree_sha256") != verification_hash
                or manifest.get("verification_file_count") != verification_count
                or manifest.get("shared_runtime_tree_sha256") != shared_runtime_hash
                or manifest.get("shared_runtime_file_count") != shared_runtime_count
            ):
                errors.append(f"{task_id}: verification dependency tree hash")
        except Exception as exc:
            errors.append(f"{task_id}: verification dependency tree: {exc}")

        declared = {
            row.get("script"): row.get("script_sha256")
            for row in manifest.get("actions", [])
            if isinstance(row, dict)
        }
        bindings = bindings_from(anchors)
        expected_scripts = {binding[1]["script"] for binding in bindings}
        if set(declared) != expected_scripts:
            errors.append(
                f"{task_id}: action manifest does not exactly cover anchor bindings"
            )

        for card, binding, kind in bindings:
            relative = binding["script"]
            script = task_assets / relative
            if (
                not script.is_file()
                or script.is_symlink()
                or sha256_file(script) != binding.get("script_sha256")
                or declared.get(relative) != binding.get("script_sha256")
            ):
                errors.append(f"{task_id}: script binding/hash: {relative}")
                continue

            environment = {
                **os.environ,
                "TASK_DATA": str((participant_task / "data").resolve()),
                "TASK_DATA_MANIFEST": str(
                    (participant_task / "data_manifest.json").resolve()
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            try:
                completed = subprocess.run(
                    [sys.executable, "-B", str(script)],
                    cwd=task_assets,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=240,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"{task_id}: action timeout: {relative}")
                continue
            action_runs += 1
            if completed.returncode:
                errors.append(
                    f"{task_id}: action failed: {relative}: {completed.stderr[-300:]}"
                )
                continue
            try:
                result = json.loads(completed.stdout)
            except Exception as exc:
                errors.append(f"{task_id}: action JSON: {relative}: {exc}")
                continue

            if kind == "perturb":
                if (
                    result.get("family") != binding.get("family")
                    or not isinstance(result.get("survive"), bool)
                ):
                    errors.append(f"{task_id}: perturbation contract: {relative}")
                continue

            quantities = {row["key"]: row for row in card.get("quantities", [])}
            for row in binding.get("values", []):
                try:
                    actual = float(value_at(result, row["json_path"]))
                    expected = float(quantities[row["quantity_key"]]["value"])
                    tolerance = float(
                        quantities[row["quantity_key"]].get("tolerance_rel", 0)
                    )
                    quantity_checks += 1
                except Exception as exc:
                    errors.append(f"{task_id}: quantity binding: {relative}: {exc}")
                    continue
                allowed = max(1e-12, abs(expected) * tolerance)
                if not math.isfinite(actual) or abs(actual - expected) > allowed:
                    errors.append(
                        f"{task_id}: quantity mismatch: {relative}:{row['quantity_key']}"
                    )

    receipt = {
        "release": "V1.0",
        "status": "PASS" if not errors else "FAIL",
        "task_count": len(registry["tasks"]),
        "action_run_count": action_runs,
        "quantity_check_count": quantity_checks,
        "novelty_card_count": novelty_cards,
        "novelty_pre_t0_hit_count": novelty_pre_t0_hits,
        "novelty_excluded_hit_count": novelty_excluded_hits,
        "errors": errors,
    }
    print(json.dumps(receipt, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-manifests",
        action="store_true",
        help="refresh complete per-task verification-tree and shared-runtime hashes",
    )
    parser.add_argument(
        "--update-novelty-hashes",
        action="store_true",
        help="refresh query-spec hashes embedded in frozen novelty assets",
    )
    arguments = parser.parse_args()
    if arguments.update_manifests:
        update_manifests()
    if arguments.update_novelty_hashes:
        update_novelty_hashes()
    raise SystemExit(main())
