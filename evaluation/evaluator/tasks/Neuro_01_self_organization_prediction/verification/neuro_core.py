#!/usr/bin/env python3
"""作者 fig3.m 的最小 Python 复刻，只计算金标所需外推量。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat


CONDITIONS = ("ctrl", "bic", "dzp", "mix0", "mix50")


def _as_trials(value: object) -> list[object]:
    array = np.asarray(value, dtype=object)
    if array.shape == ():
        return [array.item()]
    return list(array.flat)


def load_trials(data_dir: Path) -> dict[str, list[object]]:
    result: dict[str, list[object]] = {}
    for condition in CONDITIONS:
        path = data_dir / "raw" / f"response_data_{condition}.mat"
        if not path.is_file():
            path = data_dir / f"response_data_{condition}.mat"
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        result[condition] = _as_trials(payload[f"data_{condition}"])
    return result


def conditional_expectations(r: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    # MATLAB reshape(vector, [], 100)' == NumPy reshape(100, -1) for this vector order.
    def sessions(values: np.ndarray) -> np.ndarray:
        return values.reshape(100, -1).mean(axis=1)

    masks = {
        "11": (s[:, 0] == 1) & (s[:, 1] == 1),
        "10": (s[:, 0] == 1) & (s[:, 1] == 0),
        "01": (s[:, 0] == 0) & (s[:, 1] == 1),
        "00": (s[:, 0] == 0) & (s[:, 1] == 0),
    }
    output = {"all": np.column_stack([sessions(r[:, i]) for i in range(r.shape[1])])}
    for key, mask in masks.items():
        output[key] = np.column_stack(
            [sessions(r[mask, i]) for i in range(r.shape[1])]
        )
    return output


def preferred_groups(expectations: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    overall = expectations["all"]
    contrast = (expectations["10"] - expectations["01"]).mean(axis=0)
    eligible = (overall.mean(axis=0) > 1) & (overall.min(axis=0) > 0.1)
    g1 = np.flatnonzero(eligible & (contrast > 0.5))
    g2 = np.flatnonzero(eligible & (contrast < -0.5))
    if len(g1) == 1:
        g1 = np.repeat(g1, 2)
    if len(g2) == 1:
        g2 = np.repeat(g2, 2)
    if not len(g1) or not len(g2):
        raise RuntimeError("source-preferring ensemble is empty")
    return g1, g2


def baseline_excitability(
    trials: dict[str, list[object]],
) -> dict[str, np.ndarray]:
    mean_resp1: list[list[float]] = []
    mean_resp2: list[list[float]] = []
    mean_resp3: list[list[float]] = []
    mean_resp4: list[list[float]] = []
    for condition in CONDITIONS:
        for index, trial in enumerate(trials[condition]):
            r = np.asarray(trial.r, dtype=float)
            s = np.asarray(trial.s, dtype=float)
            exp = conditional_expectations(r, s)
            g1, g2 = preferred_groups(exp)
            x_mean = np.vstack(
                [exp["all"][:, g1].mean(axis=1), exp["all"][:, g2].mean(axis=1)]
            )
            pair = [float(x_mean[0, :10].mean()), float(x_mean[1, :10].mean())]
            if condition == "ctrl" and index < 23:
                mean_resp1.append(pair)
            elif condition in {"ctrl", "bic", "dzp"}:
                mean_resp2.append(pair)
            elif condition == "mix0":
                mean_resp3.append(pair)
            elif condition == "mix50":
                mean_resp4.append(pair)

    means = [
        float(np.mean(mean_resp1)),
        float(np.mean(mean_resp2)),
        float(np.mean(mean_resp3)),
        float(np.mean(mean_resp4)),
    ]
    return {
        "ctrl": np.asarray([means[0]] * 23 + [means[1]] * 7),
        "bic": np.full(len(trials["bic"]), means[1]),
        "dzp": np.full(len(trials["dzp"]), means[1]),
        "mix0": np.full(len(trials["mix0"]), means[2]),
        "mix50": np.full(len(trials["mix50"]), means[3]),
    }


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def logit(value: np.ndarray) -> np.ndarray:
    return np.log(value / (1.0 - value))


def normalize_responses(trial: object, baseline: float) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(trial.r, dtype=float)
    s = np.asarray(trial.s, dtype=float)
    o = np.asarray(trial.o, dtype=float)
    exp = conditional_expectations(r, s)
    g1, g2 = preferred_groups(exp)
    groups = (g1, g2)
    x = np.vstack([r[:, group].mean(axis=1) for group in groups])
    x_mean = np.vstack([exp["all"][:, group].mean(axis=1) for group in groups])
    condition_means = {
        key: np.vstack([exp[key][:, group].mean(axis=1) for group in groups])
        for key in ("11", "10", "01", "00")
    }
    masks = {
        "11": (s[:, 0] == 1) & (s[:, 1] == 1),
        "10": (s[:, 0] == 1) & (s[:, 1] == 0),
        "01": (s[:, 0] == 0) & (s[:, 1] == 1),
        "00": (s[:, 0] == 0) & (s[:, 1] == 0),
    }
    for key, mask in masks.items():
        x[:, mask] -= condition_means[key][:, [0]]
    x -= np.repeat(x_mean - x_mean[:, [0]], 256, axis=1)
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(
        axis=1, ddof=1, keepdims=True
    )
    x = 0.5 + x / 2.0 + (x_mean[:, [0]] - baseline) / baseline / 4.0
    return np.clip(x, 0.0, 1.0), o


def predict_trial(
    trial: object,
    baseline: float,
    *,
    ninit: int = 10,
    prior_strength: float = 3000.0,
    gain: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    x, o = normalize_responses(trial, baseline)
    nx, no, sessions = 2, 32, 100
    phi1 = np.log(x[:, : 256 * ninit].mean(axis=1))
    phi0 = np.log(1.0 - x[:, : 256 * ninit].mean(axis=1))

    w1 = np.zeros((sessions, nx, no))
    w0 = np.zeros((sessions, nx, no))
    w = np.zeros((sessions, nx, no))
    h1 = np.full((nx, no), prior_strength / 2.0)
    h0 = np.full((nx, no), prior_strength / 2.0)
    home1 = np.full((nx, no), prior_strength)
    home0 = np.full((nx, no), prior_strength)
    for t in range(sessions):
        sl = slice(256 * t, 256 * (t + 1))
        w1[t] = logit(h1 / home1)
        w0[t] = logit(h0 / home0)
        w[t] = w1[t] - w0[t]
        h1 += x[:, sl] @ o[sl]
        h0 += (1.0 - x[:, sl]) @ o[sl]
        home1 += x[:, sl].sum(axis=1)[:, None]
        home0 += (1.0 - x[:, sl]).sum(axis=1)[:, None]

    xp = np.zeros_like(x)
    w1p = np.zeros_like(w1)
    w0p = np.zeros_like(w0)
    wp = np.zeros_like(w)
    h1 = np.full((nx, no), prior_strength * gain / 2.0)
    h0 = np.full((nx, no), prior_strength * gain / 2.0)
    home1 = np.full((nx, no), prior_strength * gain)
    home0 = np.full((nx, no), prior_strength * gain)
    for t in range(sessions):
        sl = slice(256 * t, 256 * (t + 1))
        if t < ninit:
            w1p[t] = w1[t]
            w0p[t] = w0[t]
        else:
            w1p[t] = logit(h1 / home1)
            w0p[t] = logit(h0 / home0)
        wp[t] = w1p[t] - w0p[t]
        threshold1 = np.log(1.0 - sigmoid(w1p[t])).sum(axis=1) + phi1
        threshold0 = np.log(1.0 - sigmoid(w0p[t])).sum(axis=1) + phi0
        xp[:, sl] = sigmoid(o[sl] @ wp[t].T + threshold1 - threshold0).T
        update = x[:, sl] if t < ninit else xp[:, sl]
        factor = gain if t < ninit else 1.0
        h1 += update @ o[sl] * factor
        h0 += (1.0 - update) @ o[sl] * factor
        home1 += update.sum(axis=1)[:, None] * factor
        home0 += (1.0 - update).sum(axis=1)[:, None] * factor

    weight_error = np.zeros(sessions)
    response_mse = np.zeros(sessions)
    for t in range(sessions):
        sl = slice(256 * t, 256 * (t + 1))
        observed = np.vstack([sigmoid(w1[t]), sigmoid(w0[t])])
        predicted = np.vstack([sigmoid(w1p[t]), sigmoid(w0p[t])])
        weight_error[t] = np.square(observed - predicted).sum() / np.square(
            observed
        ).sum()
        response_mse[t] = np.square(x[:, sl] - xp[:, sl]).mean()
    return weight_error, response_mse


def control_metrics(data_dir: Path, *, ninit: int = 10) -> dict[str, object]:
    trials = load_trials(data_dir)
    baselines = baseline_excitability(trials)
    weight: list[float] = []
    mse: list[float] = []
    for trial, baseline in zip(trials["ctrl"], baselines["ctrl"]):
        trial_weight, trial_mse = predict_trial(trial, float(baseline), ninit=ninit)
        weight.append(float(trial_weight[99]))
        mse.append(float(trial_mse[99]))
    weight_array = np.asarray(weight)
    mse_array = np.asarray(mse)
    return {
        "ninit_sessions": ninit,
        "n_trials": len(weight),
        "session100_weight_error_mean": float(weight_array.mean()),
        "session100_weight_error_median": float(np.median(weight_array)),
        "session100_response_mse_mean": float(mse_array.mean()),
        "session100_response_score_pct": float(100.0 * (1.0 - mse_array.mean())),
        "session100_response_score_median_pct": float(
            100.0 * (1.0 - np.median(mse_array))
        ),
        "weight_errors": weight,
        "response_mse": mse,
    }


def task_data() -> Path:
    return Path(os.environ["TASK_DATA"]).resolve()
