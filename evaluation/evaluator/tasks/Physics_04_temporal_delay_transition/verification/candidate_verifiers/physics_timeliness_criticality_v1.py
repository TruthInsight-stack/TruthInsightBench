#!/usr/bin/env python3
"""Verify two clean timeliness-criticality findings and expose one excluded conflict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar


BC_THEORY = 3.99431


def phase_transition(root: Path) -> dict:
    path = root / (
        "repositories/github/jose-moran__timeliness_criticality/figures/"
        "fig2_critical_buffer/data/meanfield_data.csv"
    )
    frame = pd.read_csv(path)

    def fit(maximum_b: float) -> dict:
        selected = frame[frame["B"] <= maximum_b]
        slope, intercept = np.polyfit(selected["B"], selected["V"], 1)
        prediction = slope * selected["B"].to_numpy() + intercept
        residual = selected["V"].to_numpy() - prediction
        total = selected["V"].to_numpy() - selected["V"].mean()
        return {
            "maximum_b": maximum_b,
            "sample_count": int(len(selected)),
            "slope": float(slope),
            "intercept": float(intercept),
            "critical_buffer": float(-intercept / slope),
            "r_squared": float(1 - np.dot(residual, residual) / np.dot(total, total)),
        }

    base = fit(3.5)
    perturbations = [fit(value) for value in (3.0, 3.25)]
    above = frame[frame["B"] >= 3.75]["V"].abs().to_numpy()
    passed = (
        abs(base["critical_buffer"] - 3.674) <= 0.02
        and abs(base["slope"] + 1) <= 0.05
        and base["r_squared"] >= 0.999
        and float(np.max(above)) <= 1e-4
        and all(abs(row["critical_buffer"] - base["critical_buffer"]) <= 0.01 for row in perturbations)
    )
    return {
        "scientific_action": "fit the simulated delay velocity below the transition and verify arrest above it",
        "paper_target": {"critical_buffer": 3.674, "velocity_relation": "v approximately Bc-B below Bc"},
        "base_fit": base,
        "sample_window_perturbations": perturbations,
        "above_transition": {
            "sample_count": int(len(above)),
            "median_absolute_velocity": float(np.median(above)),
            "maximum_absolute_velocity": float(np.max(above)),
        },
        "acceptance": {"passed": passed},
    }


def correlation_collapse(root: Path) -> dict:
    path = root / (
        "repositories/github/jose-moran__timeliness_criticality/figures/"
        "fig3a_correlation_collapse/fig_source_data/inset_scatter_data.csv"
    )
    rows = pd.read_csv(path)
    b_values = rows["Bcs"].to_numpy(dtype=float)
    scales = rows["scale_factors"].to_numpy(dtype=float)
    reference_b = b_values[0]

    def model(b: np.ndarray, bc: float, exponent: float) -> np.ndarray:
        return (b - bc) ** exponent / (reference_b - bc) ** exponent

    def fit(b: np.ndarray, y: np.ndarray) -> dict:
        params, covariance = curve_fit(
            model, b, y, p0=[3.6755, 1.6936],
            bounds=([3.672, 1.0], [3.678, 2.5]), maxfev=100000,
        )
        residual = y - model(b, *params)
        return {
            "critical_buffer": float(params[0]),
            "exponent_gamma": float(params[1]),
            "covariance": covariance.tolist(),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
        }

    base = fit(b_values, scales)
    leave_one_out = []
    for index in range(1, len(b_values)):
        keep = np.arange(len(b_values)) != index
        leave_one_out.append({"dropped_index": index, **fit(b_values[keep], scales[keep])})

    def log_objective(bc: float) -> float:
        x = np.log(b_values - bc) - np.log(reference_b - bc)
        exponent = float(np.dot(x, np.log(scales)) / np.dot(x, x))
        return float(np.sum((np.log(scales) - exponent * x) ** 2))

    scalar = minimize_scalar(log_objective, bounds=(3.672, 3.678), method="bounded")
    log_x = np.log(b_values - scalar.x) - np.log(reference_b - scalar.x)
    log_exponent = float(np.dot(log_x, np.log(scales)) / np.dot(log_x, log_x))
    passed = (
        abs(base["critical_buffer"] - 3.6755) <= 0.001
        and abs(base["exponent_gamma"] - 1.6936) <= 0.08
        and abs(float(scalar.x) - base["critical_buffer"]) <= 0.001
        and abs(log_exponent - base["exponent_gamma"]) <= 0.12
    )
    return {
        "scientific_action": "fit B-dependent correlation scale factors to the paper's normalized power law",
        "paper_target": {"critical_buffer": 3.6755, "exponent_gamma": 1.6936},
        "base_curve_fit": base,
        "sample_leave_one_out": leave_one_out,
        "method_log_space": {
            "critical_buffer": float(scalar.x),
            "exponent_gamma": log_exponent,
            "objective": float(scalar.fun),
        },
        "acceptance": {"passed": passed},
    }


def excluded_finite_size_conflict(root: Path) -> dict:
    base = root / (
        "repositories/github/jose-moran__timeliness_criticality/figures/"
        "SI.2_finite_size/data"
    )
    n = np.load(base / "N_values.npy")

    def fit(path: str) -> dict:
        values = np.load(base / path)
        y = (BC_THEORY - values) ** -0.5
        design = np.column_stack([np.ones(len(n)), np.log(n)])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        return {"a": float(intercept), "b": float(slope)}

    file_label_values = {
        "tempnet_Bc.npy_labelled_STN_by_author_plot_code": fit("tempnet_Bc.npy"),
        "meanfield_Bc.npy_labelled_MF_by_author_plot_code": fit("meanfield_Bc.npy"),
    }
    paper_labels = {
        "MF": {"a": 0.4723, "b": 0.1432},
        "STN": {"a": 0.4407, "b": 0.1437},
    }
    consistent = (
        abs(file_label_values["tempnet_Bc.npy_labelled_STN_by_author_plot_code"]["a"] - paper_labels["STN"]["a"]) <= 0.01
        and abs(file_label_values["meanfield_Bc.npy_labelled_MF_by_author_plot_code"]["a"] - paper_labels["MF"]["a"]) <= 0.01
    )
    return {
        "status": "excluded_from_gold",
        "reason_code": "paper_author_code_model_label_conflict",
        "scientific_action": "compare fitted coefficients under the author file/code labels with the labels printed in Supplementary Fig. SI.2",
        "file_and_plot_code_labels": file_label_values,
        "paper_supplement_labels": paper_labels,
        "assignment_consistent": consistent,
        "note": "The two coefficient pairs are reproducible, but their MF/STN assignment is reversed between paper text and author file/plot labels; this finding is not used as a gold discovery.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discoveries = {
        "critical_buffer_phase_transition": phase_transition(args.candidate_root),
        "correlation_time_power_law": correlation_collapse(args.candidate_root),
    }
    result = {
        "schema_version": "truthinsightbench-candidate-verification",
        "candidate_id": "stagef_physics_timeliness_criticality",
        "discoveries": discoveries,
        "source_consistency_exclusions": {
            "finite_size_model_assignment": excluded_finite_size_conflict(args.candidate_root),
        },
        "passed": all(row["acceptance"]["passed"] for row in discoveries.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
