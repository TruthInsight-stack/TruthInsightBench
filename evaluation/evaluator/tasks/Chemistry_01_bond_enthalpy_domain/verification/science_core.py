#!/usr/bin/env python3
"""Frozen scientific actions for the bond-enthalpy domain task."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path


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


def _full_summary(root: Path, parity: int | None = None, unordered_fragments: bool = False) -> dict:
    row_count = 0
    molecules: set[str] = set()
    unique_bonds: set[tuple[str, str, str]] = set()
    bond_types: Counter[str] = Counter()
    path = root / "source_data" / "bde_db_full.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if parity is not None and int(row["rid"]) % 2 != parity:
                continue
            row_count += 1
            molecules.add(row["molecule"])
            fragments = (row["fragment1"], row["fragment2"])
            if unordered_fragments:
                fragments = tuple(sorted(fragments))
            unique_bonds.add((row["molecule"], *fragments))
            bond_types[row["bond_type"]] += 1
    return {
        "row_count": row_count,
        "molecule_count": len(molecules),
        "unique_bde_count": len(unique_bonds),
        "c_h_share": bond_types["C-H"] / row_count,
        "top_bond_type": bond_types.most_common(1)[0][0],
    }


def _benchmark(root: Path) -> list[dict[str, str]]:
    path = root / "normalized_views" / "experimental_benchmark.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _errors(rows: list[dict[str, str]], mode: str = "mean") -> dict[str, float]:
    observed = [float(row["experimental_bde_kcal_mol"]) for row in rows]
    output = {}
    for name, column in (
        ("quantum", "quantum_estimate_bde_kcal_mol"),
        ("learned", "learned_estimate_bde_kcal_mol"),
        ("reference", "reference_estimate_bde_kcal_mol"),
    ):
        values = [abs(float(row[column]) - target) for row, target in zip(rows, observed)]
        output[name] = statistics.mean(values) if mode == "mean" else statistics.median(values)
    return output


def summary_a(root: Path) -> dict[str, float]:
    verify_sources(root)
    result = _full_summary(root)
    return {key: result[key] for key in ("row_count", "molecule_count", "unique_bde_count", "c_h_share")}


def summary_b(root: Path) -> dict[str, float]:
    verify_sources(root)
    rows = _benchmark(root)
    errors = _errors(rows)
    learned_within_three = statistics.mean(
        abs(float(row["learned_estimate_bde_kcal_mol"]) - float(row["experimental_bde_kcal_mol"])) <= 3
        for row in rows
    )
    return {
        "quantum_mae_kcal_mol": errors["quantum"],
        "learned_mae_kcal_mol": errors["learned"],
        "reference_mae_kcal_mol": errors["reference"],
        "learned_within_3_kcal_mol_fraction": learned_within_three,
        "benchmark_observation_count": len(rows),
    }


def perturb(card_id: str, family: str, root: Path) -> dict:
    verify_sources(root)
    if card_id == "A" and family == "sample":
        halves = [_full_summary(root, parity=value) for value in (0, 1)]
        survive = all(
            row["molecule_count"] > 20000
            and row["unique_bde_count"] > 140000
            and abs(row["c_h_share"] - 0.66394) < 0.01
            for row in halves
        )
        return {"family": family, "survive": survive, "values": halves,
                "note": "deterministic even/odd record holdout"}
    if card_id == "A" and family == "method":
        result = _full_summary(root, unordered_fragments=True)
        survive = (
            result["row_count"] > result["unique_bde_count"] > result["molecule_count"]
            and result["top_bond_type"] == "C-H"
        )
        return {"family": family, "survive": survive, "values": result,
                "note": "deduplicate fragment pairs without preserving fragment order"}
    if card_id == "A" and family == "definition":
        result = _full_summary(root)
        survive = result["top_bond_type"] == "C-H" and result["c_h_share"] > 0.60
        return {"family": family, "survive": survive, "values": result,
                "note": "replace an exact share claim with the preregistered majority-domain definition"}
    if card_id == "B" and family == "sample":
        rows = _benchmark(root)
        halves = [_errors(rows[value::2]) for value in (0, 1)]
        survive = all(
            row["quantum"] < 2.3 and row["learned"] < 2.7 and row["learned"] < row["reference"] / 2
            for row in halves
        )
        return {"family": family, "survive": survive, "values": halves,
                "note": "deterministic interleaved experimental-observation holdout"}
    if card_id == "B" and family == "method":
        errors = _errors(_benchmark(root), mode="median")
        survive = errors["quantum"] < 1.6 and errors["learned"] < 1.9 and errors["learned"] < errors["reference"] / 2
        return {"family": family, "survive": survive, "values": errors,
                "note": "replace mean absolute error with median absolute error"}
    if card_id == "B" and family == "definition":
        rows = _benchmark(root)
        groups = {
            "hydrogen_fragment": [
                row for row in rows if "[H]" in (row["fragment_a_smiles"], row["fragment_b_smiles"])
            ],
            "other_fragment": [
                row for row in rows if "[H]" not in (row["fragment_a_smiles"], row["fragment_b_smiles"])
            ],
        }
        values = {name: _errors(group) for name, group in groups.items()}
        survive = all(row["learned"] < row["reference"] / 2 for row in values.values())
        return {"family": family, "survive": survive, "values": values,
                "note": "require the learned-versus-reference advantage in both bond-definition strata"}
    raise ValueError(f"unsupported perturbation: {card_id}/{family}")
