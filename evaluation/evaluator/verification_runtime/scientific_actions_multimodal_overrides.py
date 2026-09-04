#!/usr/bin/env python3
"""Task-specific corrections for multimodal scientific Actions.

The shared multimodal Action source is an immutable review dependency. Only
tasks listed here bind this additional source hash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scientific_actions_tabular import card, native, run as shared_run


ASTRONOMY_SPECTRAL_TASK = "Astronomy_05_stellar_spectral_boundary"
MULTIMODAL_OVERRIDE_TASKS = {ASTRONOMY_SPECTRAL_TASK}


def _rho(x, y) -> float:
    return float(spearmanr(np.asarray(x), np.asarray(y), nan_policy="omit").statistic)


def _corrected_stellar_card_a(data: Path) -> dict[str, Any]:
    grid = pd.read_csv(data / "photon_flux_grid.tsv", sep="\t")

    def shortwave(limit_nm: float) -> dict[str, Any]:
        rows = []
        selected = grid[grid["wavelength_nm"] <= limit_nm]
        for (temperature, wavelength), group in selected.groupby(["effective_temperature_K", "wavelength_nm"]):
            ordered = group.sort_values("metallicity_index")
            rows.append({
                "temperature": temperature,
                "wavelength": wavelength,
                "rho": _rho(ordered["metallicity_index"], ordered["photon_flux_cm2"]),
                "endpoint_ratio": ordered["photon_flux_cm2"].iloc[-1] / ordered["photon_flux_cm2"].iloc[0],
            })
        frame = pd.DataFrame(rows)
        by_temperature = frame.groupby("temperature").agg(
            median_endpoint_ratio=("endpoint_ratio", "median"),
            decreasing_fraction=("rho", lambda values: float(np.mean(values < 0))),
        )
        return {
            "wavelength_count": int(frame["wavelength"].nunique()),
            "median_endpoint_ratio": float(frame["endpoint_ratio"].median()),
            "decreasing_fraction": float(np.mean(frame["rho"] < 0)),
            "by_temperature": by_temperature.to_dict(orient="index"),
        }

    base = shortwave(200)
    cutoffs = {str(limit): shortwave(limit) for limit in (200, 220, 250)}
    temperature_holdouts = base["by_temperature"]
    return card(
        {
            "temperature_count": grid["effective_temperature_K"].nunique(),
            "shortwave_wavelength_count": base["wavelength_count"],
            "shortwave_median_high_to_low_composition_flux_ratio": base["median_endpoint_ratio"],
            "shortwave_monotone_decrease_fraction": base["decreasing_fraction"],
        },
        (
            all(row["median_endpoint_ratio"] < 0.05 and row["decreasing_fraction"] == 1 for row in temperature_holdouts.values()),
            {"complete_temperature_slices": temperature_holdouts},
        ),
        (
            all(row["median_endpoint_ratio"] < 0.05 for row in base["by_temperature"].values()),
            {"temperature_specific_endpoint_ratios": base["by_temperature"]},
        ),
        (
            all(row["median_endpoint_ratio"] < 0.20 and row["decreasing_fraction"] > 0.75 for row in cutoffs.values()),
            {"shortwave_cutoff_definitions_nm": cutoffs},
        ),
    )


def run(task_id: str, data: Path, card_id: str | None = None, family: str = "all") -> dict[str, Any]:
    result = shared_run(task_id, data)
    if task_id == ASTRONOMY_SPECTRAL_TASK:
        result["A"] = _corrected_stellar_card_a(data)
    if card_id is None:
        return native(result)
    selected = result[card_id]
    if family == "recompute":
        return {"task_id": task_id, "card_id": card_id, "values": selected["values"]}
    if family in selected["perturbations"]:
        return {"task_id": task_id, "card_id": card_id, "family": family, **selected["perturbations"][family]}
    return {"task_id": task_id, "card_id": card_id, **selected}
