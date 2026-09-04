"""Frozen evaluator Actions for anonymous variational-circuit observations."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def task_data() -> Path:
    return Path(os.environ["TASK_DATA"])


@lru_cache(maxsize=1)
def actions() -> dict[str, dict[str, Any]]:
    path = task_data() / "source_data" / "computational_observations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["action_id"]: row for row in payload["actions"]}
    required = {"rbm_exact_cost_layer", "rbm_p2_multigraph_pipeline"}
    if set(rows) != required:
        raise RuntimeError(f"unexpected computational panels: {sorted(rows)}")
    return rows


def rbm_cases() -> list[dict[str, Any]]:
    return actions()["rbm_exact_cost_layer"]["cases"]


def pipeline_cases() -> list[dict[str, Any]]:
    return actions()["rbm_p2_multigraph_pipeline"]["cases"]


def summary_a(_: Path) -> dict[str, Any]:
    cases = rbm_cases()
    return {
        "case_count": len(cases),
        "minimum_exact_state_fidelity": min(float(row["exact_state_fidelity"]) for row in cases),
        "maximum_absolute_one_minus_fidelity": max(abs(1.0 - float(row["exact_state_fidelity"])) for row in cases),
        "hidden_units_minus_edges_min": min(
            int(row["rbm_hidden_units_after_cost_layer"]) - int(row["edges"]) for row in cases
        ),
        "hidden_units_minus_edges_max": max(
            int(row["rbm_hidden_units_after_cost_layer"]) - int(row["edges"]) for row in cases
        ),
        "graph_count": len({row["graph"] for row in cases}),
    }


def summary_b(_: Path) -> dict[str, Any]:
    action = actions()["rbm_p2_multigraph_pipeline"]
    cases = pipeline_cases()
    return {
        "case_count": len(cases),
        "minimum_p1_final_fidelity": float(action["minimum_p1_final_fidelity"]),
        "minimum_compression_exact_fidelity": float(action["minimum_compression_exact_fidelity"]),
        "minimum_p2_final_fidelity": float(action["minimum_p2_final_fidelity"]),
        "minimum_gate_terminal_fidelity": float(action["minimum_gate_terminal_fidelity"]),
        "graph_seed_count": len({int(row["graph_seed"]) for row in cases}),
    }


def group_by_graph(cases: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    names = sorted({str(row["graph"]) for row in cases})
    return [(name, [row for row in cases if row["graph"] == name]) for name in names]


def perturb(card: str, family: str, _: Path) -> dict[str, Any]:
    if card == "A":
        cases = rbm_cases()
        if family == "sample":
            metrics = [
                {
                    "graph": name,
                    "case_count": len(rows),
                    "minimum_fidelity": min(row["exact_state_fidelity"] for row in rows),
                }
                for name, rows in group_by_graph(cases)
            ]
            survive = all(row["minimum_fidelity"] >= 0.9999999999 for row in metrics)
        elif family == "method":
            errors = [abs(1.0 - float(row["exact_state_fidelity"])) for row in cases]
            metrics = {
                "maximum_absolute_one_minus_fidelity": max(errors),
                "median_absolute_one_minus_fidelity": sorted(errors)[len(errors) // 2],
            }
            survive = metrics["maximum_absolute_one_minus_fidelity"] <= 1e-10
        elif family == "definition":
            offsets = [
                int(row["rbm_hidden_units_after_cost_layer"]) - int(row["edges"])
                for row in cases
            ]
            metrics = {"hidden_units_minus_edges": offsets, "unique_offsets": sorted(set(offsets))}
            survive = metrics["unique_offsets"] == [1]
        else:
            raise ValueError(family)
    elif card == "B":
        cases = pipeline_cases()
        thresholds = actions()["rbm_p2_multigraph_pipeline"][
            "frozen_thresholds_from_prior_single_graph_execution"
        ]
        if family == "sample":
            metrics = [
                {
                    "graph_seed": row["graph_seed"],
                    "p1_final_fidelity": row["p1_final_fidelity"],
                    "compression_exact_fidelity": row["compression_exact_fidelity"],
                    "p2_final_fidelity": row["p2_final_fidelity"],
                    "passes_frozen_thresholds": (
                        row["p1_final_fidelity"] >= thresholds["p1_final_fidelity"]
                        and row["compression_exact_fidelity"] >= thresholds["compression_fidelity"]
                        and row["p2_final_fidelity"] >= thresholds["p2_final_fidelity"]
                    ),
                }
                for row in cases
            ]
            survive = all(row["passes_frozen_thresholds"] for row in metrics)
        elif family == "method":
            metrics = {
                "minimum_p1_final_fidelity": min(row["p1_final_fidelity"] for row in cases),
                "minimum_compression_exact_fidelity": min(row["compression_exact_fidelity"] for row in cases),
                "minimum_p2_final_fidelity": min(row["p2_final_fidelity"] for row in cases),
            }
            survive = (
                metrics["minimum_p1_final_fidelity"] >= thresholds["p1_final_fidelity"]
                and metrics["minimum_compression_exact_fidelity"] >= thresholds["compression_fidelity"]
                and metrics["minimum_p2_final_fidelity"] >= thresholds["p2_final_fidelity"]
            )
        elif family == "definition":
            metrics = [{
                "graph_seed": row["graph_seed"],
                "p1_infidelity": abs(1.0 - row["p1_final_fidelity"]),
                "compression_infidelity": abs(1.0 - row["compression_exact_fidelity"]),
                "p2_infidelity": abs(1.0 - row["p2_final_fidelity"]),
            } for row in cases]
            survive = all(
                row["p1_infidelity"] <= 1.0 - thresholds["p1_final_fidelity"]
                and row["compression_infidelity"] <= 1.0 - thresholds["compression_fidelity"]
                and row["p2_infidelity"] <= 1.0 - thresholds["p2_final_fidelity"]
                for row in metrics
            )
        else:
            raise ValueError(family)
    else:
        raise ValueError(card)
    return {"card": card, "family": family, "survive": bool(survive), "metrics": metrics}
