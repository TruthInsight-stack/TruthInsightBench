#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


def task_data() -> Path:
    return Path(os.environ["TASK_DATA"]).resolve()


def manifest(data_dir: Path) -> list[dict[str, object]]:
    return json.loads((data_dir / "structure_manifest.json").read_text(encoding="utf-8"))[
        "structures"
    ]


def find_structure(data_dir: Path, *, r1: float, r2: float, chi: float, epsilon: float) -> Path:
    hits = [
        row
        for row in manifest(data_dir)
        if math.isclose(float(row["r1"]), r1)
        and math.isclose(float(row["r2"]), r2)
        and math.isclose(float(row["chi"]), chi)
        and math.isclose(float(row["epsilon"]), epsilon)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one structure, got {hits}")
    return data_dir / "structures" / f"{hits[0]['public_id']}.xyz"


def read_xyz(path: Path) -> tuple[dict[str, float], list[tuple[str, float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    match = re.search(
        r"R1=(?P<r1>[\d.]+), R2=(?P<r2>[\d.]+); Chi = (?P<chi>[\d.]+), "
        r"Epsilon = (?P<epsilon>[\d.]+); Energy = (?P<energy>[-\d.]+).*"
        r"Box dimension: (?P<box>[\d.]+)",
        lines[1],
    )
    if not match:
        raise RuntimeError(path)
    header = {key: float(value) for key, value in match.groupdict().items()}
    points = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        points.append((fields[0], *map(float, fields[1:4])))
    if len(points) != count:
        raise RuntimeError(f"particle count mismatch in {path}")
    return header, points


def type_z(points: list[tuple[str, float, float, float]]) -> dict[str, np.ndarray]:
    labels = sorted({point[0] for point in points})
    return {
        label: np.asarray([point[3] for point in points if point[0] == label])
        for label in labels
    }


def bilayer_metrics(data_dir: Path, *, reducer=np.mean) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, chi, epsilon in (("hex", 0.8, 0.04), ("square", 0.6, 0.08)):
        path = find_structure(data_dir, r1=3.0, r2=3.0, chi=chi, epsilon=epsilon)
        header, points = read_xyz(path)
        z = type_z(points)
        center_c = float(reducer(z["C"]))
        center_h2 = float(reducer(z["H2"]))
        output[f"{name}_abs_z_sigma"] = (abs(center_c) + abs(center_h2)) / 2.0
        output[f"{name}_preferred_z_sigma"] = header["chi"] * header["r1"]
        output[f"{name}_offset_from_preferred_sigma"] = (
            output[f"{name}_abs_z_sigma"] - output[f"{name}_preferred_z_sigma"]
        )
    return output


def distance(
    a: tuple[str, float, float, float],
    b: tuple[str, float, float, float],
    box: float,
) -> float:
    dx = abs(a[1] - b[1])
    dy = abs(a[2] - b[2])
    dx = min(dx, box - dx)
    dy = min(dy, box - dy)
    dz = a[3] - b[3]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def kagome_metrics(data_dir: Path, *, cutoff_factor: float = 1.03) -> dict[str, object]:
    path = find_structure(data_dir, r1=3.0, r2=2.5, chi=0.4, epsilon=0.08)
    header, points = read_xyz(path)
    counts = Counter(point[0] for point in points)
    z = type_z(points)
    radii = {"C": header["r1"], "H2": header["r2"]}
    c_cross_degrees: list[int] = []
    for index, point in enumerate(points):
        if point[0] != "C":
            continue
        degree = 0
        for other_index, other in enumerate(points):
            if index == other_index or other[0] != "H2":
                continue
            threshold = cutoff_factor * (radii["C"] + radii["H2"])
            if distance(point, other, header["box"]) <= threshold:
                degree += 1
        c_cross_degrees.append(degree)
    separation = float(z["H2"].mean() - z["C"].mean())
    return {
        "large_particle_count": counts["C"],
        "small_particle_count": counts["H2"],
        "finite_patch_small_large_ratio": counts["H2"] / counts["C"],
        "mean_layer_separation_sigma": separation,
        "geometric_ideal_separation_sigma": header["chi"] * (header["r1"] + header["r2"]),
        "large_to_small_contact_degree_median": float(np.median(c_cross_degrees)),
        "large_to_small_contact_degrees": c_cross_degrees,
        "cutoff_factor": cutoff_factor,
    }
