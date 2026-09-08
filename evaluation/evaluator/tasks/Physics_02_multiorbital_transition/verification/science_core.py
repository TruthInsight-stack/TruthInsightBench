#!/usr/bin/env python3
"""Frozen scientific actions for the correlation-driven transition task.

The evaluator only calls the predeclared summaries and perturbations below.  It
does not fit a new model, select a new threshold, or invent a replacement
analysis at judging time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def task_data() -> Path:
    value = os.environ.get("TASK_DATA")
    if not value:
        raise RuntimeError("TASK_DATA is required")
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(root: Path) -> None:
    manifest_path = Path(os.environ.get("TASK_DATA_MANIFEST", root.parent / "data_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"source data drift: {row['path']}")


def _rows(root: Path, name: str) -> list[dict[str, str]]:
    with (root / "normalized_views" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _gap_bracket(
    root: Path,
    filename: str,
    *,
    aggregate: str = "median",
    threshold: float = 0.0,
) -> tuple[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in _rows(root, filename):
        grouped[float(row["control_ratio"])].append(float(row["gap_like_observable"]))
    reducer = statistics.median if aggregate == "median" else statistics.mean
    summaries = {key: reducer(values) for key, values in grouped.items()}
    controls = sorted(summaries)
    upper = next(key for key in controls if summaries[key] > threshold)
    lower = max(key for key in controls if key < upper)
    return lower, upper


def summary_a(root: Path) -> dict[str, float]:
    verify_sources(root)
    a_lower, a_upper = _gap_bracket(root, "control_vs_gap_case_A.tsv")
    b_lower, b_upper = _gap_bracket(root, "control_vs_gap_case_B.tsv")
    common_lower = max(a_lower, b_lower)
    common_upper = min(a_upper, b_upper)
    return {
        "gap_case_a_onset_lower": a_lower,
        "gap_case_a_onset_upper": a_upper,
        "gap_case_b_onset_lower": b_lower,
        "gap_case_b_onset_upper": b_upper,
        "common_onset_lower": common_lower,
        "common_onset_upper": common_upper,
        "common_onset_midpoint": (common_lower + common_upper) / 2,
    }


def _topology_fraction(root: Path, control: float, parity: int | None = None) -> float:
    values: list[float] = []
    for index, row in enumerate(
        item for item in _rows(root, "control_vs_topological_proxy.tsv")
        if float(item["control_ratio"]) == control
    ):
        if parity is None or index % 2 == parity:
            values.append(float(row["degeneracy"]))
    return statistics.mean(value > 1 for value in values)


def _local_fraction(root: Path, control: float, parity: int | None = None) -> float:
    grouped: dict[float, float] = {}
    for row in _rows(root, "control_vs_local_observables.tsv"):
        grouped[float(row["control_ratio"])] = float(row["total_magnetic_moment"])
    return grouped[control] / grouped[max(grouped)]


def summary_b(root: Path) -> dict[str, float]:
    verify_sources(root)
    onset = summary_a(root)["gap_case_a_onset_upper"]
    return {
        "onset_control_ratio": onset,
        "multifold_spectrum_fraction_at_onset": _topology_fraction(root, onset),
        "local_observable_high_control_fraction_at_onset": _local_fraction(root, onset),
    }


def perturb(card_id: str, family: str, root: Path) -> dict:
    verify_sources(root)
    if card_id == "A" and family == "sample":
        brackets = {
            case: _gap_bracket(root, f"control_vs_gap_case_{case}.tsv")
            for case in ("A", "B")
        }
        survive = all(value == (0.8, 0.9) for value in brackets.values())
        return {"family": family, "survive": survive, "values": brackets,
                "note": "each independently supplied gap scan retains the same sign-change bracket"}
    if card_id == "A" and family == "method":
        brackets = {
            case: _gap_bracket(
                root, f"control_vs_gap_case_{case}.tsv", aggregate="mean"
            )
            for case in ("A", "B")
        }
        survive = all(value == (0.8, 0.9) for value in brackets.values())
        return {"family": family, "survive": survive, "values": brackets,
                "note": "replace median summaries with means"}
    if card_id == "A" and family == "definition":
        brackets = {
            case: _gap_bracket(
                root, f"control_vs_gap_case_{case}.tsv", threshold=0.002
            )
            for case in ("A", "B")
        }
        survive = all(value == (0.8, 0.9) for value in brackets.values())
        return {"family": family, "survive": survive, "values": brackets,
                "note": "require a positive margin instead of a zero crossing"}
    if card_id == "B" and family == "sample":
        rows = []
        for parity in (0, 1):
            rows.append({
                "parity": parity,
                "multifold_fraction": _topology_fraction(root, 0.9, parity),
                "local_fraction": _local_fraction(root, 0.9),
            })
        survive = all(
            row["multifold_fraction"] >= 0.75 and row["local_fraction"] < 0.65
            for row in rows
        )
        return {"family": family, "survive": survive, "values": rows,
                "note": "deterministic interleaved holdout of spectrum entries; the paired magnetic moment is not pseudo-replicated"}
    if card_id == "B" and family == "method":
        topology = [
            float(row["degeneracy"])
            for row in _rows(root, "control_vs_topological_proxy.tsv")
            if float(row["control_ratio"]) == 0.9
        ]
        rows = _rows(root, "control_vs_local_observables.tsv")
        moment = next(float(row["total_magnetic_moment"]) for row in rows if float(row["control_ratio"]) == 0.9)
        local = moment / 2.0
        values = {"median_degeneracy": statistics.median(topology), "local_theoretical_max_fraction": local}
        return {"family": family,
                "survive": values["median_degeneracy"] > 1 and local < 0.65,
                "values": values,
                "note": "use median degeneracy rather than a proportion"}
    if card_id == "B" and family == "definition":
        values = [
            {
                "control": control,
                "multifold_fraction": _topology_fraction(root, control),
                "local_fraction": _local_fraction(root, control),
            }
            for control in (0.8, 1.0)
        ]
        survive = all(
            row["multifold_fraction"] >= 0.80 and row["local_fraction"] < 0.65
            for row in values
        )
        return {"family": family, "survive": survive, "values": values,
                "note": "widen the onset definition to its two adjacent measured controls"}
    raise ValueError(f"unsupported perturbation: {card_id}/{family}")
