"""Evaluator Actions for target-specific directed-network observations."""

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
    required = {
        "directed_sparse_flower_boundary",
        "directed_disconnected_efficiency_lower_boundary",
    }
    if set(rows) != required:
        raise RuntimeError(f"unexpected observation actions: {sorted(rows)}")
    return rows


def all_structure_checks(case: dict[str, Any]) -> bool:
    return all(
        bool(value)
        for key, value in case["structure"].items()
        if key != "arc_count"
    )


def reference(action_id: str, n_nodes: int = 8, n_arcs: int = 11) -> dict[str, Any]:
    rows = [
        row for row in actions()[action_id]["cases"]
        if row["n_nodes"] == n_nodes and row["n_arcs"] == n_arcs
    ]
    if not rows:
        raise RuntimeError(f"reference case missing: {action_id}/{n_nodes}/{n_arcs}")
    return rows[0]


def summarize(action_id: str) -> dict[str, Any]:
    action = actions()[action_id]
    cases = action["cases"]
    common = {
        "case_count": len(cases),
        "maximum_absolute_efficiency_error": max(float(row["absolute_efficiency_error"]) for row in cases),
        "all_structural_checks_pass": all(all_structure_checks(row) for row in cases),
    }
    if action_id == "directed_sparse_flower_boundary":
        ref = reference(action_id)
        return {
            **common,
            "maximum_absolute_path_error": max(float(row["absolute_path_error"]) for row in cases),
            "all_cases_strongly_connected": all(bool(row["strongly_connected"]) for row in cases),
            "reference_n8_l11_path_length": float(ref["observed_path_length"]),
            "reference_n8_l11_global_efficiency": float(ref["observed_global_efficiency"]),
            "reference_n8_l11_diameter": float(ref["diameter"]),
        }
    range1 = [row for row in cases if row["construction"].startswith("range1")]
    range2 = [row for row in cases if row["construction"].startswith("range2")]
    return {
        **common,
        "range1_case_count": len(range1),
        "range2_exact_case_count": len(range2),
        "maximum_absolute_efficiency_minus_density": max(
            abs(float(row["efficiency_minus_density"])) for row in cases
        ),
        "reference_complete_dag_n8_efficiency": float(
            reference(action_id, n_nodes=8, n_arcs=28)["observed_global_efficiency"]
        ),
    }


def summary_a(_: Path) -> dict[str, Any]:
    return summarize("directed_sparse_flower_boundary")


def summary_b(_: Path) -> dict[str, Any]:
    return summarize("directed_disconnected_efficiency_lower_boundary")


def split_by_nodes(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        [row for row in cases if row["n_nodes"] == n]
        for n in sorted({int(row["n_nodes"]) for row in cases})
    ]


def numeric_pass(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return all(
        float(row.get("absolute_path_error", 0.0)) <= 1e-12
        and float(row["absolute_efficiency_error"]) <= 1e-12
        and all_structure_checks(row)
        for row in rows
    )


def perturb(card: str, family: str, _: Path) -> dict[str, Any]:
    action_id = (
        "directed_sparse_flower_boundary"
        if card == "A"
        else "directed_disconnected_efficiency_lower_boundary"
    )
    cases = actions()[action_id]["cases"]
    if family == "sample":
        groups = split_by_nodes(cases)
        metrics = [
            {"n_nodes": rows[0]["n_nodes"], "case_count": len(rows), "numeric_pass": numeric_pass(rows)}
            for rows in groups
        ]
        survive = all(row["numeric_pass"] for row in metrics)
    elif family == "method":
        if card == "A":
            metrics = [{
                "n_nodes": row["n_nodes"],
                "n_arcs": row["n_arcs"],
                "independent_vs_formula_path_error": row["absolute_path_error"],
                "independent_vs_formula_efficiency_error": row["absolute_efficiency_error"],
            } for row in cases]
            survive = numeric_pass(cases) and all(row["strongly_connected"] for row in cases)
        else:
            range1 = [row for row in cases if row["construction"].startswith("range1")]
            range2 = [row for row in cases if row["construction"].startswith("range2")]
            metrics = {
                "range1_case_count": len(range1),
                "range2_exact_case_count": len(range2),
                "range1_maximum_density_error": max(abs(row["efficiency_minus_density"]) for row in range1),
                "range2_maximum_density_error": max(abs(row["efficiency_minus_density"]) for row in range2),
            }
            survive = numeric_pass(range1) and numeric_pass(range2) and all(
                value <= 1e-12
                for key, value in metrics.items()
                if key.endswith("density_error")
            )
    elif family == "definition":
        if card == "A":
            endpoints = []
            for n_nodes in sorted({int(row["n_nodes"]) for row in cases}):
                ring = reference(action_id, n_nodes=n_nodes, n_arcs=n_nodes)
                star = reference(action_id, n_nodes=n_nodes, n_arcs=2 * (n_nodes - 1))
                endpoints.append({
                    "n_nodes": n_nodes,
                    "ring_endpoint_diameter": ring["diameter"],
                    "star_endpoint_diameter": star["diameter"],
                    "ring_is_longer_than_star": ring["observed_path_length"] > star["observed_path_length"],
                    "star_is_more_efficient_than_ring": star["observed_global_efficiency"] > ring["observed_global_efficiency"],
                })
            metrics = endpoints
            survive = all(
                row["ring_endpoint_diameter"] > 2
                and row["star_endpoint_diameter"] == 2
                and row["ring_is_longer_than_star"]
                and row["star_is_more_efficient_than_ring"]
                for row in endpoints
            )
        else:
            identities = []
            for group in split_by_nodes(cases):
                unique = {}
                for row in group:
                    unique.setdefault(int(row["n_arcs"]), row)
                ordered = [unique[key] for key in sorted(unique)]
                identities.append({
                    "n_nodes": ordered[0]["n_nodes"],
                    "efficiency_equals_density": all(
                        abs(row["observed_global_efficiency"] - row["density"]) <= 1e-12
                        for row in ordered
                    ),
                    "efficiency_nondecreasing_with_arcs": all(
                        left["observed_global_efficiency"] <= right["observed_global_efficiency"] + 1e-12
                        for left, right in zip(ordered, ordered[1:])
                    ),
                })
            metrics = identities
            survive = all(
                row["efficiency_equals_density"] and row["efficiency_nondecreasing_with_arcs"]
                for row in identities
            )
    else:
        raise ValueError(family)
    return {"card": card, "family": family, "survive": bool(survive), "metrics": metrics}
