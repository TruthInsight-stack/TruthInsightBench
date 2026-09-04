#!/usr/bin/env python3
"""Frozen scientific actions for the controlled-dynamics forecasting task."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _corr(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator


def _mae(left: list[float], right: list[float]) -> float:
    return _mean([abs(a - b) for a, b in zip(left, right)])


def _slope(control: list[float], response: list[float]) -> float:
    control_mean, response_mean = _mean(control), _mean(response)
    return sum(
        (x - control_mean) * (y - response_mean) for x, y in zip(control, response)
    ) / sum((x - control_mean) ** 2 for x in control)


def _ranks(values: list[float]) -> list[float]:
    output = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in ordered[start:stop]:
            output[index] = rank
        start = stop
    return output


FREQUENCY_FILES = (
    ("control_a_frequency_response.tsv", "control_a"),
    ("control_b_frequency_response.tsv", "control_b"),
    ("signed_control_frequency_response.tsv", "signed_control"),
)


def _frequency_rows(root: Path, filename: str) -> tuple[list[float], list[float], list[float]]:
    control_name = dict(FREQUENCY_FILES)[filename]
    rows = _rows(root, filename)
    return (
        [float(row[control_name]) for row in rows],
        [float(row["reference_frequency"]) for row in rows],
        [float(row["forecast_frequency"]) for row in rows],
    )


def _frequency_summary(root: Path, parity: int | None = None) -> list[dict[str, float]]:
    output = []
    for filename, _ in FREQUENCY_FILES:
        control, reference, forecast = _frequency_rows(root, filename)
        if parity is not None:
            control, reference, forecast = control[parity::2], reference[parity::2], forecast[parity::2]
        output.append({
            "correlation": _corr(reference, forecast),
            "mae": _mae(reference, forecast),
            "reference_slope": _slope(control, reference),
            "forecast_slope": _slope(control, forecast),
            "normalized_mae": _mae(reference, forecast) / (max(reference) - min(reference)),
            "spearman": _corr(_ranks(reference), _ranks(forecast)),
        })
    return output


def _trajectory_summary(root: Path, parity: int | None = None) -> list[dict[str, float]]:
    output = []
    for path in sorted((root / "normalized_views").glob("trajectory_condition_*.tsv")):
        rows = _rows(root, path.name)
        reference = [float(row["reference_response"]) for row in rows]
        forecast = [float(row["forecast_response"]) for row in rows]
        if parity is not None:
            reference, forecast = reference[parity::2], forecast[parity::2]
        rmse = math.sqrt(_mean([(a - b) ** 2 for a, b in zip(reference, forecast)]))
        output.append({
            "correlation": _corr(reference, forecast),
            "nrmse": rmse / statistics.pstdev(reference),
            "mae": _mae(reference, forecast),
        })
    return output


def summary_a(root: Path) -> dict[str, float]:
    verify_sources(root)
    rows = _frequency_summary(root)
    return {
        "minimum_frequency_correlation": min(row["correlation"] for row in rows),
        "maximum_frequency_mae": max(row["mae"] for row in rows),
        "matching_slope_direction_count": sum(
            (row["reference_slope"] > 0) == (row["forecast_slope"] > 0) for row in rows
        ),
        "control_scan_count": len(rows),
    }


def summary_b(root: Path) -> dict[str, float]:
    verify_sources(root)
    rows = _trajectory_summary(root)
    output: dict[str, float] = {}
    for index, row in enumerate(rows, start=1):
        output[f"condition_{index}_correlation"] = row["correlation"]
        output[f"condition_{index}_nrmse"] = row["nrmse"]
    return output


def perturb(card_id: str, family: str, root: Path) -> dict:
    verify_sources(root)
    if card_id == "A" and family == "sample":
        halves = [_frequency_summary(root, parity=value) for value in (0, 1)]
        survive = all(
            min(row["correlation"] for row in half) > 0.95
            and max(row["mae"] for row in half) < 0.15
            and all((row["reference_slope"] > 0) == (row["forecast_slope"] > 0) for row in half)
            for half in halves
        )
        return {"family": family, "survive": survive, "values": halves,
                "note": "deterministic interleaved holdout within every complete control scan"}
    if card_id == "A" and family == "method":
        rows = _frequency_summary(root)
        values = [row["spearman"] for row in rows]
        return {"family": family, "survive": min(values) > 0.95, "values": values,
                "note": "replace Pearson agreement with rank agreement"}
    if card_id == "A" and family == "definition":
        rows = _frequency_summary(root)
        values = [row["normalized_mae"] for row in rows]
        return {"family": family, "survive": max(values) < 0.10, "values": values,
                "note": "normalize absolute error by each measured response span"}
    if card_id == "B" and family == "sample":
        halves = [_trajectory_summary(root, parity=value) for value in (0, 1)]
        survive = all(
            rows[0]["correlation"] > rows[1]["correlation"] > rows[2]["correlation"]
            and rows[0]["nrmse"] < rows[1]["nrmse"] < rows[2]["nrmse"]
            for rows in halves
        )
        return {"family": family, "survive": survive, "values": halves,
                "note": "interleaved time-point holdout within every full trajectory"}
    if card_id == "B" and family == "method":
        rows = _trajectory_summary(root)
        values = [row["mae"] for row in rows]
        return {"family": family, "survive": values[0] < values[1] < values[2], "values": values,
                "note": "replace correlation/NRMSE with raw mean absolute error"}
    if card_id == "B" and family == "definition":
        rows = _trajectory_summary(root)
        passing = sum(row["correlation"] >= 0.80 for row in rows)
        return {"family": family, "survive": passing == 2,
                "values": {"high_fidelity_condition_count": passing},
                "note": "apply a preregistered correlation threshold to locate the fidelity boundary"}
    raise ValueError(f"unsupported perturbation: {card_id}/{family}")
