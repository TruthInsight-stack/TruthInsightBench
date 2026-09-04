"""Frozen evaluator Actions for anonymous learned network-dynamics data."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve()
PACKAGED_RUNTIME = HERE.parent / "author_runtime"
AUTHOR_ROOT = PACKAGED_RUNTIME if PACKAGED_RUNTIME.is_dir() else HERE.parents[1] / "author_code" / "source"
MODEL_FOLDER = (
    "results/er_experiment_MAK_0_size_100_std_reg_1.0_self_int_True_"
    "nbr_int_True_self_hidden_1_nbr_hidden_1_single_gnnlayer_True"
)
EXPECTED = {
    "er_n_100_p_01.npy": "d9b5123e8a09e273524991d28f988a07197fa7a4e2b3230a70bf04914f841c8d",
    "er_n_100_p_01_alt.npy": "4f8c6604ff870ce43667e3affe1da01dc7971f7acd552ca29363d2e3435ca1d8",
    "er_n_100_p_06.npy": "ad03d570f668538e16afb936135f2ddc5423ecf09be9c3c46e3868c12d414ffd",
}
BASE_SEED = 20260813
SHIFT = 0.2
SIGNIFICANCE_ALPHA = 0.001
GRAPH_FILES = {
    "training_graph": "er_n_100_p_01.npy",
    "equivalent_novel_graph": "er_n_100_p_01_alt.npy",
    "higher_degree_graph": "er_n_100_p_06.npy",
}


def task_data() -> Path:
    return Path(os.environ["TASK_DATA"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_runtime_imports() -> None:
    # The archived author utility imports torchdiffeq although load_results does
    # not call it.  A failing stub makes that unused dependency explicit while
    # keeping the frozen load path self-contained.
    if "torchdiffeq" not in sys.modules:
        module = types.ModuleType("torchdiffeq")

        def unused_odeint(*_: Any, **__: Any) -> Any:
            raise RuntimeError("torchdiffeq.odeint is outside this frozen Action")

        module.odeint = unused_odeint  # type: ignore[attr-defined]
        sys.modules["torchdiffeq"] = module
    sys.path.insert(0, str(AUTHOR_ROOT))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


@lru_cache(maxsize=1)
def loaded() -> dict[str, Any]:
    if not AUTHOR_ROOT.is_dir():
        raise RuntimeError(f"frozen author runtime missing: {AUTHOR_ROOT}")
    install_runtime_imports()
    from utilities import load_results  # type: ignore
    from dynamics import Dynamics  # type: ignore

    source = task_data() / "source_data"
    for name, expected in EXPECTED.items():
        path = source / name
        if sha256(path) != expected:
            raise RuntimeError(f"graph input hash drifted: {name}")
    model_dir = source / MODEL_FOLDER
    torch.set_num_threads(1)
    torch.manual_seed(BASE_SEED)
    torch.use_deterministic_algorithms(True)
    adjacency, training, dynamics_config, _, model, _, x_train, y_train, x_test, y_test = load_results(str(model_dir))
    model.eval()
    if len(adjacency) != 1 or len(x_train) != 900 or len(x_train) != len(y_train):
        raise RuntimeError("persisted author model/data structure drifted")
    return {
        "source": source,
        "adjacency": adjacency,
        "training": training,
        "dynamics_config": dynamics_config,
        "model": model,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "Dynamics": Dynamics,
    }


def relative_percent_error(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    denominator = torch.sum(torch.abs(truth))
    if float(denominator) == 0.0:
        raise RuntimeError("zero denominator in relative-percent error")
    return float(torch.sum(torch.abs(prediction - truth)) / denominator * 100.0)


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_percent": float(array.mean()),
        "median_percent": float(np.median(array)),
        "p95_percent": float(np.quantile(array, 0.95)),
        "max_percent": float(array.max()),
    }


@lru_cache(maxsize=1)
def stored_errors() -> tuple[float, ...]:
    item = loaded()
    model, adjacency = item["model"], item["adjacency"]
    values = []
    with torch.no_grad():
        for state, truth in zip(item["x_train"], item["y_train"]):
            prediction = model(None, state[:, None], adjacency[0]).squeeze()
            values.append(relative_percent_error(prediction, truth.squeeze()))
    return tuple(values)


def sample_values(rng: np.random.Generator, distribution: str, size: int) -> np.ndarray:
    if distribution == "uniform_0_1":
        return rng.uniform(0.0, 1.0, (size, 1))
    if distribution == "beta_5_2":
        return rng.beta(5.0, 2.0, (size, 1))
    raise ValueError(distribution)


def generated_panel(graph_name: str, distribution: str, seed: int, count: int) -> dict[str, Any]:
    item = loaded()
    matrix = torch.as_tensor(np.load(item["source"] / graph_name), dtype=torch.float32)
    config = item["dynamics_config"]
    exact = item["Dynamics"](
        matrix,
        model=config.model_name,
        B=config.B,
        R=config.R,
        H=config.H,
        F=config.F,
        a=config.a,
        b=config.b,
    )
    rng = np.random.default_rng(seed)
    values = []
    with torch.no_grad():
        for _ in range(count):
            state = torch.as_tensor(sample_values(rng, distribution, matrix.shape[0]), dtype=torch.float32)
            truth = exact(0, state).squeeze()
            prediction = item["model"](None, state[:, None], matrix).squeeze()
            values.append(relative_percent_error(prediction, truth))
    return {
        "graph": graph_name,
        "distribution": distribution,
        "seed": seed,
        "node_count": int(matrix.shape[0]),
        "mean_degree": float(matrix.sum() / matrix.shape[0]),
        **summarize(values),
    }


@lru_cache(maxsize=1)
def generalization_panels() -> dict[str, dict[str, Any]]:
    return {
        "stored_training_pairs": summarize(list(stored_errors())),
        "training_graph_uniform": generated_panel("er_n_100_p_01.npy", "uniform_0_1", 20260813, 1000),
        "equivalent_graph_uniform": generated_panel("er_n_100_p_01_alt.npy", "uniform_0_1", 20260913, 1000),
        "equivalent_graph_beta_5_2": generated_panel("er_n_100_p_01_alt.npy", "beta_5_2", 20260914, 1000),
        "higher_degree_graph_uniform": generated_panel("er_n_100_p_06.npy", "uniform_0_1", 20261013, 1000),
        "higher_degree_graph_beta_5_2": generated_panel("er_n_100_p_06.npy", "beta_5_2", 20261014, 1000),
    }


@lru_cache(maxsize=1)
def domain_boundary_panels() -> dict[str, dict[str, Any]]:
    item = loaded()
    source, config = item["source"], item["dynamics_config"]
    panels: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for graph_index, (graph_id, graph_file) in enumerate(GRAPH_FILES.items()):
            matrix = torch.as_tensor(np.load(source / graph_file), dtype=torch.float32)
            exact = item["Dynamics"](
                matrix,
                model=config.model_name,
                B=config.B,
                R=config.R,
                H=config.H,
                F=config.F,
                a=config.a,
                b=config.b,
            )
            seed = 20263013 + graph_index * 100
            rng = np.random.default_rng(seed)
            in_domain_errors = []
            shifted_errors = []
            for _ in range(1000):
                base_values = rng.uniform(0.0, 1.0, (matrix.shape[0], 1))
                pair = []
                for values in (base_values, base_values + SHIFT):
                    state = torch.as_tensor(values, dtype=torch.float32)
                    truth = exact(0, state).squeeze()
                    prediction = item["model"](None, state[:, None], matrix).squeeze()
                    pair.append(relative_percent_error(prediction, truth))
                in_domain_errors.append(pair[0])
                shifted_errors.append(pair[1])
            in_domain = np.asarray(in_domain_errors, dtype=np.float64)
            shifted = np.asarray(shifted_errors, dtype=np.float64)
            differences = shifted - in_domain
            test = wilcoxon(shifted, in_domain, alternative="greater", zero_method="wilcox", method="auto")
            panels[graph_id] = {
                "graph_file": graph_file,
                "node_count": int(matrix.shape[0]),
                "mean_degree": float(matrix.sum() / matrix.shape[0]),
                "paired_seed": seed,
                "pair_count": len(in_domain_errors),
                "in_domain_uniform_0_1": summarize(in_domain_errors),
                "out_of_domain_uniform_0_2_1_2": summarize(shifted_errors),
                "mean_error_increase_percent_points": float(differences.mean()),
                "median_error_increase_percent_points": float(np.median(differences)),
                "mean_error_ratio": float(shifted.mean() / in_domain.mean()),
                "fraction_pairs_shifted_error_greater": float(np.mean(differences > 0)),
                "paired_wilcoxon_statistic": float(test.statistic),
                "paired_wilcoxon_one_sided_p_value": float(test.pvalue),
                "_in_domain_errors": in_domain_errors,
                "_shifted_errors": shifted_errors,
            }
    return panels


def projection() -> dict[str, Any]:
    path = task_data() / "source_data" / "computational_observations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summary_a(_: Path) -> dict[str, Any]:
    panels = generalization_panels()
    archived = projection()["actions"][0]
    for name, row in panels.items():
        if abs(row["mean_percent"] - archived["panels"][name]["mean_percent"]) > 1e-12:
            raise RuntimeError(f"generalization replay disagrees with archived execution: {name}")
    means = [row["mean_percent"] for row in panels.values()]
    p95s = [row["p95_percent"] for row in panels.values()]
    return {
        "panel_count": len(panels),
        "maximum_panel_mean_percent": max(means),
        "maximum_panel_p95_percent": max(p95s),
        "panel_mean_spread_percent_points": max(means) - min(means),
    }


def summary_b(_: Path) -> dict[str, Any]:
    panels = domain_boundary_panels()
    archived = projection()["actions"][1]
    for name, row in panels.items():
        if abs(row["mean_error_ratio"] - archived["panels"][name]["mean_error_ratio"]) > 1e-12:
            raise RuntimeError(f"domain-boundary replay disagrees with archived execution: {name}")
    ratios = [row["mean_error_ratio"] for row in panels.values()]
    p_values = [row["paired_wilcoxon_one_sided_p_value"] for row in panels.values()]
    return {
        "graph_panel_count": len(panels),
        "minimum_mean_error_ratio": min(ratios),
        "maximum_one_sided_p_value": max(p_values),
        "all_graphs_directionally_worse": all(
            row["mean_error_increase_percent_points"] > 0
            and row["median_error_increase_percent_points"] > 0
            for row in panels.values()
        ),
    }


def perturb(card: str, family: str, _: Path) -> dict[str, Any]:
    if card == "A":
        panels = generalization_panels()
        if family == "sample":
            groups = {
                "training": [panels["stored_training_pairs"], panels["training_graph_uniform"]],
                "equivalent_unseen_graph": [panels["equivalent_graph_uniform"], panels["equivalent_graph_beta_5_2"]],
                "higher_degree_unseen_graph": [panels["higher_degree_graph_uniform"], panels["higher_degree_graph_beta_5_2"]],
            }
            rows = [{
                "topology_group": name,
                "panel_count": len(values),
                "maximum_mean_percent": max(row["mean_percent"] for row in values),
                "maximum_p95_percent": max(row["p95_percent"] for row in values),
            } for name, values in groups.items()]
            survive = all(row["maximum_mean_percent"] < 1.0 and row["maximum_p95_percent"] < 1.0 for row in rows)
        elif family == "method":
            rows = [{
                "panel": name,
                "mean_percent": row["mean_percent"],
                "median_percent": row["median_percent"],
                "p95_percent": row["p95_percent"],
            } for name, row in panels.items()]
            survive = all(
                row["mean_percent"] < 1.0
                and row["median_percent"] < 1.0
                and row["p95_percent"] < 1.0
                for row in rows
            )
        elif family == "definition":
            source = loaded()["source"]
            training = np.load(source / "er_n_100_p_01.npy")
            equivalent = np.load(source / "er_n_100_p_01_alt.npy")
            degree_difference = abs(
                float(equivalent.sum() / equivalent.shape[0])
                - float(training.sum() / training.shape[0])
            ) / float(training.sum() / training.shape[0])
            rows = [{
                "training_input_support": [0.0, 1.0],
                "uniform_test_support": [0.0, 1.0],
                "beta_5_2_test_support": [0.0, 1.0],
                "equivalent_graph_mean_degree_relative_difference": degree_difference,
                "all_panel_means_below_one_percent": all(row["mean_percent"] < 1.0 for row in panels.values()),
            }]
            survive = degree_difference <= 0.05 and rows[0]["all_panel_means_below_one_percent"]
        else:
            raise ValueError(family)
    elif card == "B":
        panels = domain_boundary_panels()
        if family == "sample":
            rows = [{
                "graph": name,
                "mean_error_ratio": row["mean_error_ratio"],
                "fraction_pairs_shifted_error_greater": row["fraction_pairs_shifted_error_greater"],
                "one_sided_p_value": row["paired_wilcoxon_one_sided_p_value"],
            } for name, row in panels.items()]
            survive = all(
                row["mean_error_ratio"] > 1.0
                and row["fraction_pairs_shifted_error_greater"] >= 0.95
                and row["one_sided_p_value"] < SIGNIFICANCE_ALPHA
                for row in rows
            )
        elif family == "method":
            rows = []
            for name, panel in panels.items():
                base_values = np.asarray(panel["_in_domain_errors"], dtype=np.float64)
                shifted_values = np.asarray(panel["_shifted_errors"], dtype=np.float64)
                for block_index, indices in enumerate(np.array_split(np.arange(len(base_values)), 4)):
                    test = wilcoxon(shifted_values[indices], base_values[indices], alternative="greater")
                    rows.append({
                        "graph": name,
                        "block": block_index,
                        "pair_count": len(indices),
                        "mean_difference": float(np.mean(shifted_values[indices] - base_values[indices])),
                        "one_sided_p_value": float(test.pvalue),
                    })
            survive = all(
                row["mean_difference"] > 0 and row["one_sided_p_value"] < SIGNIFICANCE_ALPHA
                for row in rows
            )
        elif family == "definition":
            rows = [{
                "graph": name,
                "training_domain": [0.0, 1.0],
                "shifted_domain": [0.2, 1.2],
                "median_error_ratio": (
                    row["out_of_domain_uniform_0_2_1_2"]["median_percent"]
                    / row["in_domain_uniform_0_1"]["median_percent"]
                ),
                "median_error_increase_percent_points": row["median_error_increase_percent_points"],
            } for name, row in panels.items()]
            survive = all(
                row["median_error_ratio"] > 1.0
                and row["median_error_increase_percent_points"] > 0
                for row in rows
            )
        else:
            raise ValueError(family)
    else:
        raise ValueError(card)
    return {"card": card, "family": family, "survive": bool(survive), "metrics": rows}
