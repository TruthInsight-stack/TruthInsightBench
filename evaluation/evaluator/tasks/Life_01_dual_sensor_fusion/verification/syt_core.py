#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def task_data() -> Path:
    return Path(os.environ["TASK_DATA"]).resolve()


def fusion_runs(data_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    path = data_dir / "derived_for_navigation/fusion_dose_independent_runs.csv"
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[(row["condition"].strip(), row["metric"])].append(
                float(row["value_pct"])
            )
    return {key: np.asarray(items) for key, items in values.items()}


def first_frame_runs(data_dir: Path) -> dict[str, np.ndarray]:
    # source_table_01 的 0.147 s 第一帧，每组取 5 个独立 run。
    path = data_dir / "source_tables/source_table_01.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return {
        "None": np.asarray([float(x) for x in rows[20][2:7]]),
        "1:200 SensorB": np.asarray([float(x) for x in rows[20][47:52]]),
    }


def binding_runs(data_dir: Path) -> dict[float, np.ndarray]:
    path = data_dir / "derived_for_navigation/binding_competition_runs.csv"
    values: dict[float, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[float(row["competitor_concentration_uM"])].append(
                float(row["bound_target_uM"])
            )
    return {key: np.asarray(items) for key, items in values.items()}


def fusion_metrics(data_dir: Path, *, reducer=np.mean) -> dict[str, float]:
    runs = fusion_runs(data_dir)
    total_none = float(reducer(runs[("None", "total_fusion")]))
    total_high = float(reducer(runs[("1:200 SensorB", "total_fusion")]))
    early_none = float(reducer(runs[("None", "fusion_first_two_frames")]))
    early_high = float(reducer(runs[("1:200 SensorB", "fusion_first_two_frames")]))
    return {
        "total_fusion_none_pct": total_none,
        "total_fusion_high_pct": total_high,
        "total_fusion_change_pp": total_high - total_none,
        "early_fusion_none_pct": early_none,
        "early_fusion_high_pct": early_high,
        "early_fusion_drop_pp": early_none - early_high,
    }


def binding_metrics(
    data_dir: Path, *, high_concentration: float = 30.0, reducer=np.mean
) -> dict[str, float]:
    runs = binding_runs(data_dir)
    zero = float(reducer(runs[0.0]))
    high = float(reducer(runs[high_concentration]))
    return {
        "binding_zero_uM": zero,
        "binding_30uM_uM": high,
        "binding_drop_pct": 100.0 * (zero - high) / zero,
    }
