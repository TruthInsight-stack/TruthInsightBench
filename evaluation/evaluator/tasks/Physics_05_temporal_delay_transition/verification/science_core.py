"""Frozen evaluator Actions for anonymous temporal-delay transition data."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
VERIFIER_ROOT = HERE.parent / "candidate_verifiers"
sys.path.insert(0, str(VERIFIER_ROOT))
import physics_timeliness_criticality_v1 as verifier  # noqa: E402


def task_data() -> Path:
    return Path(os.environ["TASK_DATA"])


@lru_cache(maxsize=1)
def phase() -> dict[str, Any]:
    # The frozen verifier only joins this constant path. A path proxy keeps
    # its scientific implementation unchanged without duplicating data.
    return _run_rebased(verifier.phase_transition)


def _run_rebased(function):
    # Reproduce the verifier's documented path mapping without copying data.
    class RootProxy:
        def __truediv__(self, raw: str):
            prefix = "repositories/github/jose-moran__timeliness_criticality/"
            if not raw.startswith(prefix):
                raise RuntimeError(raw)
            return task_data() / "source_data" / raw[len(prefix):]
    return function(RootProxy())


@lru_cache(maxsize=1)
def collapse() -> dict[str, Any]:
    return _run_rebased(verifier.correlation_collapse)


def summary_a(_: Path) -> dict[str, Any]:
    row = phase()
    return {
        "critical_buffer": row["base_fit"]["critical_buffer"],
        "velocity_slope": row["base_fit"]["slope"],
        "r_squared": row["base_fit"]["r_squared"],
        "post_transition_max_abs_velocity": row["above_transition"]["maximum_absolute_velocity"],
    }


def summary_b(_: Path) -> dict[str, Any]:
    row = collapse()["base_curve_fit"]
    return {"critical_buffer": row["critical_buffer"], "exponent_gamma": row["exponent_gamma"], "rmse": row["rmse"]}


def perturb(card: str, family: str, _: Path) -> dict[str, Any]:
    if card == "A":
        row = phase()
        if family == "sample":
            metrics = {"fit_windows": row["sample_window_perturbations"]}
            base = row["base_fit"]["critical_buffer"]
            survive = all(abs(item["critical_buffer"] - base) <= 0.01 for item in metrics["fit_windows"])
        elif family == "method":
            metrics = {"base_fit": row["base_fit"]}
            survive = abs(metrics["base_fit"]["slope"] + 1.0) <= 0.05 and metrics["base_fit"]["r_squared"] >= 0.999
        elif family == "definition":
            metrics = {"above_transition": row["above_transition"]}
            survive = metrics["above_transition"]["maximum_absolute_velocity"] <= 1e-4
        else: raise ValueError(family)
    elif card == "B":
        row = collapse()
        if family == "sample":
            metrics = {"leave_one_out": row["sample_leave_one_out"]}
            survive = all(3.674 <= item["critical_buffer"] <= 3.677 and 1.55 <= item["exponent_gamma"] <= 1.90 for item in metrics["leave_one_out"])
        elif family == "method":
            metrics = {"nonlinear_fit": row["base_curve_fit"], "log_space_fit": row["method_log_space"]}
            survive = abs(metrics["nonlinear_fit"]["critical_buffer"] - metrics["log_space_fit"]["critical_buffer"]) <= 0.001 and abs(metrics["nonlinear_fit"]["exponent_gamma"] - metrics["log_space_fit"]["exponent_gamma"]) <= 0.12
        elif family == "definition":
            metrics = {"base": row["base_curve_fit"]}
            survive = 3.674 <= metrics["base"]["critical_buffer"] <= 3.677 and 1.55 <= metrics["base"]["exponent_gamma"] <= 1.85
        else: raise ValueError(family)
    else: raise ValueError(card)
    return {"card": card, "family": family, "survive": bool(survive), "metrics": metrics}
