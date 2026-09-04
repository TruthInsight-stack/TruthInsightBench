#!/usr/bin/env python3
"""Narrow, hash-bound corrections for task-specific scientific actions.

The established shared action modules are immutable review dependencies: changing
one of them would invalidate every task that already binds its hash.  Corrections
that affect only one task therefore live here and are explicitly listed in that
task's action manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scientific_actions_tabular import run as shared_run


ASTRONOMY_TASK = "Astronomy_04_gravity_spectral_structure"


def _astronomy_comparable_information_criteria(data: Path) -> dict[str, float]:
    """Compare absolute criteria only; the ranked grid is delta-scaled."""
    structured = pd.read_csv(data / "candidate_spectrum_comparison.tsv", sep="\t")
    uniform = pd.read_csv(data / "uniform_alternative_comparison.tsv", sep="\t")
    best_structured = float(structured["information_criterion"].min())
    best_uniform = float(uniform["information_criterion"].min())
    return {
        "best_structured_slice_information_criterion": best_structured,
        "best_uniform_information_criterion": best_uniform,
        "information_criterion_gap": round(best_uniform - best_structured, 10),
    }


def run(
    task_id: str,
    data: Path,
    card_id: str | None = None,
    family: str = "all",
) -> dict[str, Any]:
    """Dispatch to shared actions and correct the Astronomy Card-B comparator."""
    result = shared_run(task_id, data, card_id, family)
    if task_id != ASTRONOMY_TASK:
        return result

    corrected = _astronomy_comparable_information_criteria(data)
    if card_id is None:
        result["B"]["values"].update(corrected)
    elif card_id == "B" and "values" in result:
        result["values"].update(corrected)
    return result
