#!/usr/bin/env python3
"""EarthScience_01 的冻结复算与扰动核心。

只读取公开题包中的两张主表；所有统计定义在赛前冻结，不由 Judge 临时生成。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_SHA256 = {
    "glacial_erosion_Earth.tsv": "b49192d77082fd431963ca67278d9face1079e8054711b84296554573c24e2cf",
    "nonglacial_erosion_Earth.tsv": "214063142d936c3a5408583b682397042392b4d46e0443e30c88811fbe9fee89",
}
RATE = "Erosion rate (mm/yr)"
TIME = "Time interval (yr)"


def task_data() -> Path:
    value = os.environ.get("TASK_DATA")
    if not value:
        raise RuntimeError("TASK_DATA is required")
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root: Path, filename: str) -> pd.DataFrame:
    path = root / "raw" / filename
    actual = sha256(path)
    expected = SOURCE_SHA256[filename]
    if actual != expected:
        raise RuntimeError(f"source data drift: {filename}: {actual} != {expected}")
    frame = pd.read_csv(path, sep="\t")
    frame["rate"] = pd.to_numeric(frame[RATE], errors="coerce")
    frame["time"] = pd.to_numeric(frame[TIME], errors="coerce")
    return frame[(frame["rate"] > 0) & (frame["time"] > 0)].copy()


def datasets(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glacial = _load(root, "glacial_erosion_Earth.tsv")
    nonglacial = _load(root, "nonglacial_erosion_Earth.tsv")
    fluvial = nonglacial[nonglacial["Type"] == "Fluvial"].copy()
    subaerial = nonglacial[nonglacial["Type"] == "Subaerial"].copy()
    return glacial, fluvial, subaerial


def geometric_mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(array))))


def loglog_slope(frame: pd.DataFrame) -> float:
    return float(np.polyfit(np.log10(frame["time"]), np.log10(frame["rate"]), 1)[0])


def summary_a(root: Path) -> dict[str, float]:
    glacial, fluvial, subaerial = datasets(root)
    glacial_mean = geometric_mean(glacial["rate"])
    fluvial_mean = geometric_mean(fluvial["rate"])
    return {
        "glacial_log_mean_mm_per_year": glacial_mean,
        "fluvial_log_mean_mm_per_year": fluvial_mean,
        "subaerial_log_mean_mm_per_year": geometric_mean(subaerial["rate"]),
        "glacial_to_fluvial_ratio": glacial_mean / fluvial_mean,
        "glacial_n": float(len(glacial)),
        "fluvial_n": float(len(fluvial)),
        "subaerial_n": float(len(subaerial)),
    }


def summary_b(root: Path) -> dict[str, float]:
    glacial, fluvial, subaerial = datasets(root)
    combined = pd.concat([glacial, fluvial, subaerial], ignore_index=True)
    cosmogenic = combined[
        combined["Methodology"].astype(str).str.startswith("Cosmogenic")
    ]
    thickness = cosmogenic["rate"] * cosmogenic["time"]
    return {
        "cosmogenic_loglog_slope": loglog_slope(cosmogenic),
        "cosmogenic_geometric_thickness_mm": geometric_mean(thickness),
        "cosmogenic_median_thickness_mm": float(np.median(thickness)),
        "glacial_loglog_slope": loglog_slope(glacial),
        "fluvial_loglog_slope": loglog_slope(fluvial),
        "cosmogenic_n": float(len(cosmogenic)),
    }


def _leave_one_compilation_ratios(
    glacial: pd.DataFrame, fluvial: pd.DataFrame
) -> list[dict[str, float | str]]:
    compilations = sorted(
        set(glacial["Compilation"].dropna().astype(str))
        | set(fluvial["Compilation"].dropna().astype(str))
    )
    output = []
    for compilation in compilations:
        g = glacial[glacial["Compilation"].astype(str) != compilation]
        f = fluvial[fluvial["Compilation"].astype(str) != compilation]
        if len(g) and len(f):
            output.append(
                {
                    "omitted_compilation": compilation,
                    "ratio": geometric_mean(g["rate"]) / geometric_mean(f["rate"]),
                }
            )
    return output


def perturb(card_id: str, family: str, root: Path) -> dict:
    glacial, fluvial, subaerial = datasets(root)
    combined = pd.concat([glacial, fluvial, subaerial], ignore_index=True)
    cosmogenic = combined[
        combined["Methodology"].astype(str).str.startswith("Cosmogenic")
    ]

    if card_id == "A" and family == "definition":
        ratio = float(np.median(glacial["rate"]) / np.median(fluvial["rate"]))
        return {
            "family": family,
            "survive": ratio > 1.0,
            "values": {"median_glacial_to_fluvial_ratio": ratio},
            "note": "replace log means with medians; glacial remains faster",
        }
    if card_id == "A" and family == "method":
        g = glacial[
            ~glacial["Methodology"].astype(str).str.startswith("Cosmogenic")
        ]
        f = fluvial[
            ~fluvial["Methodology"].astype(str).str.startswith("Cosmogenic")
        ]
        ratio = geometric_mean(g["rate"]) / geometric_mean(f["rate"])
        return {
            "family": family,
            "survive": ratio > 1.0,
            "values": {
                "noncosmogenic_glacial_to_fluvial_ratio": ratio,
                "glacial_n": len(g),
                "fluvial_n": len(f),
            },
            "note": "remove both cosmogenic methods before comparing rates",
        }
    if card_id == "A" and family == "sample":
        rows = _leave_one_compilation_ratios(glacial, fluvial)
        minimum = min(float(row["ratio"]) for row in rows)
        return {
            "family": family,
            "survive": minimum > 1.0,
            "values": {
                "leave_one_compilation_min_ratio": minimum,
                "leave_one_compilation_max_ratio": max(float(row["ratio"]) for row in rows),
                "omission_count": len(rows),
            },
            "note": "omit each source compilation in turn",
        }

    if card_id == "B" and family == "definition":
        detrital = cosmogenic[cosmogenic["Methodology"] == "Cosmogenic detrital"]
        surface = cosmogenic[cosmogenic["Methodology"] == "Cosmogenic surface"]
        values = {
            "detrital_slope": loglog_slope(detrital),
            "surface_slope": loglog_slope(surface),
        }
        return {
            "family": family,
            "survive": values["detrital_slope"] < -0.85 and values["surface_slope"] < -0.75,
            "values": values,
            "note": "split cosmogenic measurements into detrital and surface definitions",
        }
    if card_id == "B" and family == "method":
        volumetric = combined[combined["Methodology"] == "Volumetric"]
        values = {
            "cosmogenic_slope": loglog_slope(cosmogenic),
            "volumetric_slope": loglog_slope(volumetric),
        }
        return {
            "family": family,
            "survive": values["cosmogenic_slope"] < -0.85 and values["volumetric_slope"] > -0.2,
            "values": values,
            "note": "the near-minus-one relation is specific to cosmogenic measurement, not volumetric erosion",
        }
    if card_id == "B" and family == "sample":
        rows = []
        for compilation in sorted(cosmogenic["Compilation"].dropna().astype(str).unique()):
            sample = cosmogenic[cosmogenic["Compilation"].astype(str) != compilation]
            if len(sample) > 20:
                rows.append(
                    {
                        "omitted_compilation": compilation,
                        "slope": loglog_slope(sample),
                    }
                )
        minimum = min(float(row["slope"]) for row in rows)
        maximum = max(float(row["slope"]) for row in rows)
        return {
            "family": family,
            "survive": minimum > -1.1 and maximum < -0.8,
            "values": {
                "leave_one_compilation_min_slope": minimum,
                "leave_one_compilation_max_slope": maximum,
                "omission_count": len(rows),
            },
            "note": "omit each cosmogenic source compilation in turn",
        }
    raise ValueError(f"unsupported perturbation: {card_id}/{family}")
