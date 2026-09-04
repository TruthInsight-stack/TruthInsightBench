#!/usr/bin/env python3
"""Frozen scientific actions for the anonymous radio-burst task."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


REFERENCE_SERIES = ("population_series_1", "population_series_2")
GROUP_B_SERIES = ("population_series_3", "population_series_4")
ALTERNATE_SERIES = (
    "population_series_3",
    "population_series_4",
    "population_series_5",
    "population_series_6",
)


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


def _composition(root: Path) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"slow": 0.0, "fast": 0.0})
    for row in _rows(root, "drift_population_series.tsv"):
        totals[row["population_series_id"]][row["drift_regime"]] += float(row["event_count"])
    return dict(totals)


def _aggregate(composition: dict[str, dict[str, float]], ids: tuple[str, ...]) -> dict[str, float]:
    slow = sum(composition[series_id]["slow"] for series_id in ids)
    fast = sum(composition[series_id]["fast"] for series_id in ids)
    return {
        "slow_count": slow,
        "fast_count": fast,
        "fast_fraction": fast / (slow + fast),
        "slow_to_fast_ratio": slow / fast,
    }


def _event_rates(root: Path) -> dict[str, float]:
    return {
        row["event_id"]: abs(float(row["reported_event_drift_rate_MHz_s"]))
        for row in _rows(root, "dynamic_spectrum_event_metadata.tsv")
    }


def summary_a(root: Path) -> dict[str, float]:
    verify_sources(root)
    composition = _composition(root)
    reference = _aggregate(composition, REFERENCE_SERIES)
    alternate = _aggregate(composition, ALTERNATE_SERIES)
    return {
        "reference_fast_fraction": reference["fast_fraction"],
        "alternate_fast_fraction": alternate["fast_fraction"],
        "fast_fraction_difference": reference["fast_fraction"] - alternate["fast_fraction"],
        "reference_slow_to_fast_ratio": reference["slow_to_fast_ratio"],
        "alternate_slow_to_fast_ratio": alternate["slow_to_fast_ratio"],
    }


def summary_b(root: Path) -> dict[str, float]:
    verify_sources(root)
    group_b = _aggregate(_composition(root), GROUP_B_SERIES)
    events = _event_rates(root)
    slow_rate = events["event_3"]
    fast_rate = events["event_4"]
    return {
        "group_b_slow_fraction": 1.0 - group_b["fast_fraction"],
        "group_b_fast_fraction": group_b["fast_fraction"],
        "group_b_slow_example_abs_mhz_s": slow_rate,
        "group_b_fast_example_abs_mhz_s": fast_rate,
        "group_b_example_rate_ratio": fast_rate / slow_rate,
    }


def _leave_one_population_series(root: Path) -> list[dict[str, float | str]]:
    composition = _composition(root)
    rows: list[dict[str, float | str]] = []
    for omitted in REFERENCE_SERIES + ALTERNATE_SERIES:
        reference_ids = tuple(value for value in REFERENCE_SERIES if value != omitted)
        alternate_ids = tuple(value for value in ALTERNATE_SERIES if value != omitted)
        reference = _aggregate(composition, reference_ids)
        alternate = _aggregate(composition, alternate_ids)
        rows.append({
            "omitted": omitted,
            "reference_fast_fraction": reference["fast_fraction"],
            "alternate_fast_fraction": alternate["fast_fraction"],
            "difference": reference["fast_fraction"] - alternate["fast_fraction"],
        })
    return rows


def _event_definition_check(root: Path, threshold: float) -> dict[str, float]:
    rates = _event_rates(root)
    reference = [rates["event_1"], rates["event_2"]]
    alternate = [rates["event_3"], rates["event_4"], rates["event_5"]]
    return {
        "threshold_abs_mhz_s": threshold,
        "reference_fast_fraction": sum(value >= threshold for value in reference) / len(reference),
        "alternate_fast_fraction": sum(value >= threshold for value in alternate) / len(alternate),
    }


def perturb(card_id: str, family: str, root: Path) -> dict:
    verify_sources(root)
    composition = _composition(root)
    if card_id == "A" and family == "sample":
        values = _leave_one_population_series(root)
        survive = all(
            row["reference_fast_fraction"] > 0.85
            and row["alternate_fast_fraction"] < 0.35
            and row["difference"] > 0.55
            for row in values
        )
        return {"family": family, "survive": survive, "values": values,
                "note": "leave out each complete polarization/source population series"}
    if card_id == "A" and family == "method":
        shares = {
            series_id: _aggregate(composition, (series_id,))["fast_fraction"]
            for series_id in REFERENCE_SERIES + ALTERNATE_SERIES
        }
        values = {
            "reference_macro_fast_fraction": statistics.mean(shares[value] for value in REFERENCE_SERIES),
            "alternate_macro_fast_fraction": statistics.mean(shares[value] for value in ALTERNATE_SERIES),
        }
        values["difference"] = values["reference_macro_fast_fraction"] - values["alternate_macro_fast_fraction"]
        return {"family": family, "survive": values["difference"] > 0.60,
                "values": values, "note": "replace pooled counts with equal-weight series means"}
    if card_id == "A" and family == "definition":
        values = [_event_definition_check(root, threshold) for threshold in (8.0, 12.0)]
        survive = all(
            row["reference_fast_fraction"] == 1.0
            and row["alternate_fast_fraction"] <= 1 / 3
            for row in values
        )
        return {"family": family, "survive": survive, "values": values,
                "note": "replace source histogram labels with two event-level absolute-rate thresholds"}
    if card_id == "B" and family == "sample":
        values = []
        for omitted in GROUP_B_SERIES:
            kept = tuple(value for value in GROUP_B_SERIES if value != omitted)
            result = _aggregate(composition, kept)
            values.append({"omitted": omitted, **result})
        survive = all(0.05 < row["fast_fraction"] < 0.40 for row in values)
        return {"family": family, "survive": survive, "values": values,
                "note": "leave out each complete polarization population series in anonymous group B"}
    if card_id == "B" and family == "method":
        channel_shares = [
            _aggregate(composition, (series_id,))["fast_fraction"]
            for series_id in GROUP_B_SERIES
        ]
        values = {
            "macro_fast_fraction": statistics.mean(channel_shares),
            "minimum_channel_fast_fraction": min(channel_shares),
            "maximum_channel_fast_fraction": max(channel_shares),
        }
        return {"family": family,
                "survive": 0.05 < values["minimum_channel_fast_fraction"] and values["maximum_channel_fast_fraction"] < 0.40,
                "values": values, "note": "replace pooled counts with equal-weight polarization means"}
    if card_id == "B" and family == "definition":
        events = _event_rates(root)
        values = [
            {
                "threshold_abs_mhz_s": threshold,
                "slow_example_is_slow": events["event_3"] < threshold,
                "fast_example_is_fast": events["event_4"] >= threshold,
            }
            for threshold in (8.0, 12.0)
        ]
        return {"family": family,
                "survive": all(row["slow_example_is_slow"] and row["fast_example_is_fast"] for row in values),
                "values": values, "note": "vary the event-level fast/slow threshold"}
    raise ValueError(f"unsupported perturbation: {card_id}/{family}")
