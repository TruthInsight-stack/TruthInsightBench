#!/usr/bin/env python3
"""Executable scientific verification Actions over the task data directory."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress, spearmanr, theilslopes


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def card(values: dict, sample: tuple[bool, dict], method: tuple[bool, dict], definition: tuple[bool, dict]) -> dict:
    return {
        "values": native(values),
        "perturbations": {
            "sample": {"survive": bool(sample[0]), "metrics": native(sample[1])},
            "method": {"survive": bool(method[0]), "metrics": native(method[1])},
            "definition": {"survive": bool(definition[0]), "metrics": native(definition[1])},
        },
    }


def astronomy(data: Path) -> dict:
    traces = pd.read_csv(data / "pass_residual_traces.tsv", sep="\t")
    a = traces["residual_model_A_mm_s"].to_numpy()
    b = traces["residual_model_B_mm_s"].to_numpy()
    rms_a = np.sqrt(np.mean(a**2))
    rms_b = np.sqrt(np.mean(b**2))
    pass_ratios = {
        name: np.sqrt(np.mean(group["residual_model_B_mm_s"] ** 2) / np.mean(group["residual_model_A_mm_s"] ** 2))
        for name, group in traces.groupby("pass_id")
    }
    mae_ratio = np.mean(np.abs(b)) / np.mean(np.abs(a))
    median_abs_ratio = np.median(np.abs(b)) / np.median(np.abs(a))
    window_ratios = {}
    for width in (10, 20, 30, 40):
        group = traces[np.abs(traces["relative_time_min"]) <= width]
        window_ratios[str(width)] = np.sqrt(
            np.mean(group["residual_model_B_mm_s"] ** 2) / np.mean(group["residual_model_A_mm_s"] ** 2)
        )
    out_a = card(
        {
            "pass_count": traces["pass_id"].nunique(),
            "numeric_row_count": len(traces),
            "model_A_rms_mm_s": rms_a,
            "model_B_rms_mm_s": rms_b,
            "rms_ratio_B_over_A": rms_b / rms_a,
            "complete_pass_improvement_fraction": np.mean(np.asarray(list(pass_ratios.values())) < 1),
            "maximum_complete_pass_rms_ratio": max(pass_ratios.values()),
        },
        (max(pass_ratios.values()) < 1, {"all_22_whole_pass_rms_ratios": pass_ratios}),
        (mae_ratio < 1 and median_abs_ratio < 1, {"mae_ratio": mae_ratio, "median_absolute_ratio": median_abs_ratio}),
        (max(window_ratios.values()) < 1, {"absolute_time_window_rms_ratios": window_ratios}),
    )

    candidates = pd.read_csv(data / "candidate_spectrum_comparison.tsv", sep="\t")
    ranked = pd.read_csv(data / "ranked_full_parameter_grid.tsv", sep="\t")
    uniform = pd.read_csv(data / "uniform_alternative_comparison.tsv", sep="\t")
    modes = pd.read_csv(data / "mode_frequency_grid.tsv", sep="\t")
    best = ranked.loc[ranked["delta_information_criterion"].idxmin()]
    best_uniform = uniform.loc[uniform["information_criterion"].idxmin()]
    family_best = {
        name: group.loc[group["information_criterion"].idxmin()].to_dict()
        for name, group in candidates.groupby("candidate_family")
    }
    near = ranked[ranked["delta_information_criterion"] <= 20]
    family_gaps = {
        name: best_uniform["information_criterion"] - row["information_criterion"]
        for name, row in family_best.items()
    }
    cutoff_regions = {}
    for delta in (10, 20, 50):
        group = ranked[ranked["delta_information_criterion"] <= delta]
        cutoff_regions[str(delta)] = {
            "frequency_min": group["peak_frequency_microhz"].min(),
            "frequency_max": group["peak_frequency_microhz"].max(),
            "velocity_min": group["max_velocity_cm_s"].min(),
            "velocity_max": group["max_velocity_cm_s"].max(),
        }
    top_holdouts = {}
    for omit in range(min(10, len(ranked))):
        retained = ranked.drop(ranked.index[omit])
        row = retained.loc[retained["delta_information_criterion"].idxmin()]
        top_holdouts[str(omit + 1)] = {
            "peak_frequency_microhz": row["peak_frequency_microhz"],
            "max_velocity_cm_s": row["max_velocity_cm_s"],
            "delta_information_criterion": row["delta_information_criterion"],
        }
    nearest_mode_offset = np.min(np.abs(modes["mode_frequency_microhz"] - best["peak_frequency_microhz"]))
    out_b = card(
        {
            "best_structured_delta_information_criterion": best["delta_information_criterion"],
            "best_uniform_information_criterion": best_uniform["information_criterion"],
            "information_criterion_gap": best_uniform["information_criterion"] - best["delta_information_criterion"],
            "best_peak_frequency_microhz": best["peak_frequency_microhz"],
            "best_max_velocity_cm_s": best["max_velocity_cm_s"],
            "best_min_velocity_cm_s": best["min_velocity_cm_s"],
            "best_frequency_width_microhz": best["frequency_width_microhz"],
            "delta20_frequency_min_microhz": near["peak_frequency_microhz"].min(),
            "delta20_frequency_max_microhz": near["peak_frequency_microhz"].max(),
        },
        (
            all(900 <= row["peak_frequency_microhz"] <= 1300 for row in top_holdouts.values()),
            {"leave_one_top_ranked_candidate": top_holdouts},
        ),
        (
            all(900 <= row["frequency_min"] <= row["frequency_max"] <= 1300 for row in cutoff_regions.values()),
            {"near_optimal_regions_by_delta_information_criterion": cutoff_regions},
        ),
        (
            min(family_gaps.values()) > 1_000 and nearest_mode_offset < 10,
            {"supporting_slice_gaps_vs_uniform": family_gaps, "supporting_slice_best": family_best, "nearest_mode_offset_microhz": nearest_mode_offset},
        ),
    )
    return {"A": out_a, "B": out_b}


def chemistry(data: Path) -> dict:
    coordinate_metadata = json.loads((data / "coordinate_definitions.json").read_text(encoding="utf-8"))
    angle_definition = coordinate_metadata["angle_coordinate"]
    hbar_as_eV = coordinate_metadata["phase_delay_conversion"]["hbar_as_eV"]
    projection = pd.read_csv(
        data / "angle_energy_observations_datasets.tsv",
        sep="\t",
        keep_default_na=False,
        dtype={"dataset_path": "string", "index_0": "int64", "index_1": "string", "value": "float64"},
        low_memory=False,
    )

    def projected_array(name: str) -> np.ndarray:
        rows = projection[projection["dataset_path"] == name]
        if rows.empty:
            raise KeyError(f"Missing projected dataset: {name}")
        index_0 = rows["index_0"].to_numpy(dtype=int)
        values = rows["value"].to_numpy(dtype=float)
        index_1_text = rows["index_1"].astype(str)
        if bool((index_1_text == "").all()):
            result = np.empty(int(index_0.max()) + 1, dtype=float)
            result[index_0] = values
            return result
        index_1 = index_1_text.to_numpy(dtype=int)
        result = np.empty((int(index_0.max()) + 1, int(index_1.max()) + 1), dtype=float)
        result[index_0, index_1] = values
        return result

    panel = {}
    for name in ("experiment", "theory"):
        energy = projected_array(f"{name}/photon_energy_eV_del")
        delay = projected_array(f"{name}/delay")
        angle = np.arange(delay.shape[1], dtype=float) * angle_definition["step_deg"] + angle_definition["range_deg"][0]
        panel[name] = (energy, delay, angle)

    def cone_contrast(delay_row, angle, width=30):
        values = [
            np.median(delay_row[angle <= width]),
            np.median(delay_row[(angle >= 90 - width / 2) & (angle <= 90 + width / 2)]),
            np.median(delay_row[angle >= 180 - width]),
        ]
        return max(values) - min(values)

    values_a = {}
    window_metrics = {}
    method_metrics = {}
    definition_metrics = {}
    for name, (energy, delay, angle) in panel.items():
        index = int(np.argmin(np.abs(energy - 31.0)))
        values_a[f"{name}_energy_eV"] = energy[index]
        values_a[f"{name}_cone_contrast_as"] = cone_contrast(delay[index], angle)
        values_a[f"{name}_q90_q10_spread_as"] = np.percentile(delay[index], 90) - np.percentile(delay[index], 10)
        window_metrics[name] = []
        for target in (30.0, 30.5, 31.0, 31.5, 32.0):
            row = int(np.argmin(np.abs(energy - target)))
            window_metrics[name].append({"energy_eV": energy[row], "cone_contrast_as": cone_contrast(delay[row], angle)})
        method_metrics[name] = {
            "q90_q10_spread_as": np.percentile(delay[index], 90) - np.percentile(delay[index], 10),
            "cone_contrast_as": cone_contrast(delay[index], angle),
        }
        definition_metrics[name] = {
            str(width): cone_contrast(delay[index], angle, width) for width in (20, 30, 40)
        }
    out_a = card(
        values_a,
        (
            min(item["cone_contrast_as"] for rows in window_metrics.values() for item in rows) > 180,
            {"energy_window_contrasts": window_metrics},
        ),
        (
            min(row["q90_q10_spread_as"] for row in method_metrics.values()) > 200,
            {"quantile_and_cone_summaries": method_metrics},
        ),
        (
            min(value for rows in definition_metrics.values() for value in rows.values()) > 150,
            {"alternative_cone_widths_deg": definition_metrics},
        ),
    )

    def forward_relation(energy, phase, delay, limit):
        estimate = np.diff(np.unwrap(phase, axis=0), axis=0) / np.diff(energy)[:, None] * hbar_as_eV
        x = estimate.ravel()
        y = delay.ravel()
        keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < limit) & (np.abs(y) < limit)
        return {
            "correlation": np.corrcoef(x[keep], y[keep])[0, 1],
            "median_absolute_error_as": np.median(np.abs(x[keep] - y[keep])),
            "rows": int(np.sum(keep)),
        }

    base = {}
    thirds = {}
    central = {}
    limits = {}
    for name in ("experiment", "theory"):
        energy = projected_array(f"{name}/photon_energy_eV")
        phase = projected_array(f"{name}/phase")
        delay = projected_array(f"{name}/delay")
        base[name] = forward_relation(energy, phase, delay, 1_000)
        cuts = np.linspace(0, len(delay), 4, dtype=int)
        thirds[name] = []
        estimate = np.diff(np.unwrap(phase, axis=0), axis=0) / np.diff(energy)[:, None] * hbar_as_eV
        for start, end in zip(cuts[:-1], cuts[1:]):
            x = estimate[start:end].ravel()
            y = delay[start:end].ravel()
            keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1_000) & (np.abs(y) < 1_000)
            thirds[name].append(np.corrcoef(x[keep], y[keep])[0, 1])
        gradient = np.gradient(np.unwrap(phase, axis=0), energy, axis=0)[:-1] * hbar_as_eV
        x = gradient.ravel()
        y = delay.ravel()
        keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1_000) & (np.abs(y) < 1_000)
        central[name] = np.corrcoef(x[keep], y[keep])[0, 1]
        limits[name] = {str(limit): forward_relation(energy, phase, delay, limit)["correlation"] for limit in (500, 1_000, 2_000)}
    out_b = card(
        {
            "experiment_phase_delay_correlation": base["experiment"]["correlation"],
            "theory_phase_delay_correlation": base["theory"]["correlation"],
            "experiment_median_absolute_error_as": base["experiment"]["median_absolute_error_as"],
            "theory_median_absolute_error_as": base["theory"]["median_absolute_error_as"],
        },
        (min(value for rows in thirds.values() for value in rows) > 0.999, {"contiguous_energy_third_correlations": thirds}),
        (min(central.values()) > 0.995, {"central_gradient_correlations": central}),
        (min(value for rows in limits.values() for value in rows.values()) > 0.999, {"finite_delay_cutoff_correlations": limits}),
    )
    return {"A": out_a, "B": out_b}


def earth_particles(data: Path) -> dict:
    env = pd.read_csv(data / "sample_environment_profiles.tsv", sep="\t")
    composition = pd.read_csv(data / "particle_composition_profiles.tsv", sep="\t")
    medians = env.groupby("collection_group_id")["particle_total_per_L"].median()
    leave_ratios = {name: medians.drop(name).max() / medians.drop(name).min() for name in medians.index}
    geometric = env.groupby("collection_group_id")["particle_total_per_L"].apply(lambda x: np.exp(np.mean(np.log(x))))
    uncertainty_filters = {}
    relative_error = env["particle_total_std_or_error_per_L"] / env["particle_total_per_L"]
    for threshold in (0.5, 1.0, 2.0):
        selected = env[relative_error <= threshold].groupby("collection_group_id")["particle_total_per_L"].median()
        uncertainty_filters[str(threshold)] = selected.max() / selected.min()
    out_a = card(
        {
            "sample_count": len(env),
            "collection_group_count": env["collection_group_id"].nunique(),
            "maximum_to_minimum_group_median_ratio": medians.max() / medians.min(),
            "maximum_to_minimum_sample_ratio": env["particle_total_per_L"].max() / env["particle_total_per_L"].min(),
        },
        (min(leave_ratios.values()) > 5, {"leave_one_group_median_ratios": leave_ratios}),
        (geometric.max() / geometric.min() > 5, {"group_geometric_means": geometric.to_dict()}),
        (min(uncertainty_filters.values()) > 10, {"relative_uncertainty_filter_ratios": uncertainty_filters}),
    )

    combined = composition["polymer_PE_percent"] + composition["polymer_varnish_percent"]
    polymer_columns = [name for name in composition.columns if name.startswith("polymer_") and name.endswith("_percent")]
    dominant = composition[polymer_columns].idxmax(axis=1)
    dominant_pair = dominant.isin(["polymer_PE_percent", "polymer_varnish_percent"])
    leave_combined = {
        name: np.median(combined[composition["collection_group_id"] != name])
        for name in composition["collection_group_id"].unique()
    }
    group_medians = composition.assign(combined=combined).groupby("collection_group_id")["combined"].median()
    thresholds = {str(value): int(np.sum(combined > value)) for value in (50, 60, 75)}
    out_b = card(
        {
            "combined_PE_varnish_median_percent": np.median(combined),
            "PE_or_varnish_dominant_sample_count": int(np.sum(dominant_pair)),
            "total_sample_count": len(composition),
            "PE_or_varnish_dominant_fraction": np.mean(dominant_pair),
        },
        (min(leave_combined.values()) > 70, {"leave_one_group_combined_medians_percent": leave_combined}),
        (np.median(group_medians) > 70, {"group_equal_medians_percent": group_medians.to_dict()}),
        (thresholds["50"] >= 24 and thresholds["60"] >= 22, {"samples_above_combined_share_threshold": thresholds}),
    )
    return {"A": out_a, "B": out_b}


def earth_events(data: Path) -> dict:
    frame = pd.read_csv(data / "observations.tsv", sep="\t")
    sulfate = frame["c04_sulfate_ug_kg"].to_numpy()
    event = frame["c07_signal"].notna() & (frame["c07_signal"] != "reference_surface")
    event_indices = np.flatnonzero(event)

    def local_metrics(width):
        ratios = []
        percentiles = []
        for index in event_indices:
            start = max(0, index - width)
            end = min(len(sulfate), index + width + 1)
            local = sulfate[start:end]
            background = np.delete(local, index - start)
            ratios.append(sulfate[index] / np.nanmedian(background))
            percentiles.append(np.nanmean(local <= sulfate[index]))
        return {
            "median_event_to_local_background_ratio": np.nanmedian(ratios),
            "minimum_event_to_local_background_ratio": np.nanmin(ratios),
            "exact_local_maximum_fraction": np.mean(np.isclose(percentiles, 1.0)),
            "median_local_percentile": np.median(percentiles),
        }

    windows = {str(width): local_metrics(width) for width in (5, 10, 20, 30)}
    identified = frame["c07_signal"].str.startswith("event_", na=False)
    unidentified = frame["c07_signal"] == "unidentified_event"
    subgroup = {}
    for name, mask in (("identified", identified), ("unidentified", unidentified)):
        values = frame.loc[mask, "c04_sulfate_ug_kg"]
        subgroup[name] = {"count": len(values), "median_sulfate": values.median()}
    base_a = local_metrics(10)
    out_a = card(
        {
            "event_count": len(event_indices),
            "event_median_sulfate": np.nanmedian(sulfate[event]),
            "background_median_sulfate": np.nanmedian(sulfate[~event]),
            **base_a,
        },
        (
            all(row["minimum_event_to_local_background_ratio"] > 1.25 for row in windows.values()),
            {"contiguous_local_windows": windows},
        ),
        (base_a["median_local_percentile"] >= 0.95, {"rank_based_local_peak_summary": base_a}),
        (subgroup["identified"]["count"] >= 5 and subgroup["unidentified"]["count"] >= 8, {"event_definition_subgroups": subgroup}),
    )

    quartile = pd.qcut(frame["c06_depth_w_e_m"], 4, labels=False, duplicates="drop")
    medians = [np.nanmedian(frame.loc[quartile == index, "c04_sulfate_ug_kg"]) for index in range(4)]
    upper = [np.nanpercentile(frame.loc[quartile == index, "c04_sulfate_ug_kg"], 95) for index in range(4)]
    no_event = frame[~event].copy()
    no_event_quartile = pd.qcut(no_event["c06_depth_w_e_m"], 4, labels=False, duplicates="drop")
    no_event_medians = [np.nanmedian(no_event.loc[no_event_quartile == index, "c04_sulfate_ug_kg"]) for index in range(4)]
    third_ratios = {}
    third = pd.qcut(frame["c06_depth_w_e_m"], 3, labels=False, duplicates="drop")
    for index in range(3):
        values = frame.loc[third == index, "c04_sulfate_ug_kg"]
        third_ratios[str(index)] = np.nanpercentile(values, 95) / np.nanmedian(values)
    out_b = card(
        {
            "depth_quartile_median_ratio": max(medians) / min(medians),
            "depth_quartile_median_cv": np.std(medians) / np.mean(medians),
            "depth_quartile_upper_tail_ratio": max(upper) / min(upper),
        },
        (max(no_event_medians) / min(no_event_medians) < 1.10, {"event_excluded_quartile_medians": no_event_medians}),
        (np.std(medians) / np.mean(medians) < 0.05, {"robust_quartile_medians": medians, "upper_95th_percentiles": upper}),
        (min(third_ratios.values()) > 1.5, {"third_block_upper_tail_to_median_ratios": third_ratios}),
    )
    return {"A": out_a, "B": out_b}


def photocatalysis(data: Path) -> dict:
    curves = pd.read_csv(data / "conversion_timecourses.tsv", sep="\t")

    def slopes(frame):
        return {
            name: linregress(group["illumination_time_h"], group["gaseous_product_umol"]).slope
            for name, group in frame.groupby("formulation_id")
        }

    base_slopes = slopes(curves)
    time_sets = ((0, 1, 2), (2, 3, 4), (0, 2, 4))
    window_slopes = {}
    rankings = []
    for times in time_sets:
        selected = curves[curves["illumination_time_h"].isin(times)]
        result = slopes(selected)
        window_slopes["_".join(map(str, times))] = result
        rankings.append(sorted(result, key=result.get, reverse=True))
    endpoint_rates = {}
    for name, group in curves.groupby("formulation_id"):
        group = group.sort_values("illumination_time_h")
        endpoint_rates[name] = (group.iloc[-1]["gaseous_product_umol"] - group.iloc[0]["gaseous_product_umol"]) / (
            group.iloc[-1]["illumination_time_h"] - group.iloc[0]["illumination_time_h"]
        )
    expected_rank = sorted(base_slopes, key=base_slopes.get, reverse=True)
    robust_slopes = {
        name: theilslopes(group["gaseous_product_umol"], group["illumination_time_h"]).slope
        for name, group in curves.groupby("formulation_id")
    }
    robust_rank = sorted(robust_slopes, key=robust_slopes.get, reverse=True)
    out_a = card(
        {
            "maximum_slope_umol_per_h": max(base_slopes.values()),
            "minimum_slope_umol_per_h": min(base_slopes.values()),
            "maximum_to_minimum_slope_ratio": max(base_slopes.values()) / min(base_slopes.values()),
            "formulation_rank": expected_rank,
        },
        (all(row == expected_rank for row in rankings), {"whole_time_window_slopes": window_slopes}),
        (max(endpoint_rates.values()) / min(endpoint_rates.values()) > 40, {"endpoint_rates": endpoint_rates}),
        (robust_rank == expected_rank, {"theil_sen_slopes": robust_slopes, "theil_sen_rank": robust_rank}),
    )

    spectra = pd.read_csv(data / "optical_action_spectra.tsv", sep="\t")
    absorption = spectra[spectra["measurement_role"] == "absorption"].sort_values("wavelength_nm")
    action = spectra[spectra["measurement_role"] == "action_response"].sort_values("wavelength_nm")
    x = np.interp(action["wavelength_nm"], absorption["wavelength_nm"], absorption["response_au_or_percent"])
    y = action["response_au_or_percent"].to_numpy()
    fit = linregress(x, y)
    prediction = fit.intercept + fit.slope * x
    action_peak_index = int(np.argmax(y))
    leave_one = []
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        leave_one.append(np.corrcoef(x[keep], y[keep])[0, 1])
    windows = {}
    wavelengths = action["wavelength_nm"].to_numpy()
    for low, high in ((400, 550), (450, 600), (500, 650), (400, 600), (450, 650)):
        keep = (wavelengths >= low) & (wavelengths <= high)
        windows[f"{low}_{high}"] = np.corrcoef(x[keep], y[keep])[0, 1]
    rho = spearmanr(x, y).statistic
    out_b = card(
        {
            "pearson_correlation": np.corrcoef(x, y)[0, 1],
            "spearman_correlation": rho,
            "linear_fit_r_squared": fit.rvalue**2,
            "range_normalized_rmse": np.sqrt(np.mean((y - prediction) ** 2)) / (y.max() - y.min()),
            "peak_action_wavelength_nm": wavelengths[action_peak_index],
            "peak_action_response_percent": y[action_peak_index],
        },
        (min(leave_one) > 0.99, {"leave_one_wavelength_pearson": leave_one}),
        (rho > 0.90, {"rank_correlation": rho}),
        (min(windows.values()) > 0.99, {"contiguous_wavelength_window_pearson": windows}),
    )
    return {"A": out_a, "B": out_b}


def flow_battery(data: Path) -> dict:
    summary = pd.read_csv(data / "rate_tradeoff_summaries.tsv", sep="\t")
    efficiency = summary[summary["summary_role"] == "efficiency"].sort_values("anonymous_rate_value")
    power = summary[summary["summary_role"] == "power"].sort_values("anonymous_rate_value")
    voltage = pd.read_csv(data / "rate_dependent_voltage_curves.tsv", sep="\t")

    def gaps(low, high):
        result = {}
        for setting in sorted(voltage["anonymous_rate_setting"].unique()):
            charge = voltage[(voltage["anonymous_rate_setting"] == setting) & (voltage["scan_role"] == "charge_like")].sort_values("state_of_charge_percent")
            discharge = voltage[(voltage["anonymous_rate_setting"] == setting) & (voltage["scan_role"] == "discharge_like")].sort_values("state_of_charge_percent")
            grid = np.linspace(max(low, charge["state_of_charge_percent"].min(), discharge["state_of_charge_percent"].min()), min(high, charge["state_of_charge_percent"].max(), discharge["state_of_charge_percent"].max()), 101)
            gap = np.interp(grid, charge["state_of_charge_percent"], charge["cell_voltage_V"]) - np.interp(grid, discharge["state_of_charge_percent"], discharge["cell_voltage_V"])
            result[setting] = np.median(gap)
        return result

    base_gaps = gaps(10, 90)
    peak_power = power.loc[power["response_value"].idxmax()]
    leave_efficiency = []
    for index in efficiency.index:
        group = efficiency.drop(index)
        leave_efficiency.append(spearmanr(group["anonymous_rate_value"], group["response_value"]).statistic)
    gap_windows = {f"{low}_{high}": gaps(low, high) for low, high in ((10, 40), (30, 70), (60, 90))}
    out_a = card(
        {
            "efficiency_spearman_vs_rate": spearmanr(efficiency["anonymous_rate_value"], efficiency["response_value"]).statistic,
            "efficiency_first_percent": efficiency.iloc[0]["response_value"],
            "efficiency_last_percent": efficiency.iloc[-1]["response_value"],
            "peak_power_rate": peak_power["anonymous_rate_value"],
            "peak_power_value": peak_power["response_value"],
            "highest_rate_power_to_peak_ratio": power.iloc[-1]["response_value"] / peak_power["response_value"],
            "first_to_last_median_voltage_gap_ratio": list(base_gaps.values())[-1] / list(base_gaps.values())[0],
        },
        (max(leave_efficiency) <= -0.99, {"leave_one_efficiency_spearman": leave_efficiency}),
        (
            all(np.all(np.diff(list(row.values())) > 0) for row in gap_windows.values()),
            {"state_window_median_voltage_gaps": gap_windows},
        ),
        (peak_power["anonymous_rate_value"] < power.iloc[-1]["anonymous_rate_value"] and power.iloc[-1]["response_value"] < peak_power["response_value"], {"power_boundary": {"peak": peak_power.to_dict(), "highest_rate": power.iloc[-1].to_dict()}}),
    )

    cycles = pd.read_csv(data / "long_cycle_capacity.tsv", sep="\t")
    x = cycles["cycle_index"].to_numpy()
    y = cycles["capacity_like_value"].to_numpy()
    fit = linregress(x, y)
    segment_slopes = {}
    for start, end in ((1, 600), (101, 600), (301, 600)):
        keep = (x >= start) & (x <= end)
        segment_slopes[f"{start}_{end}"] = linregress(x[keep], y[keep]).slope
    robust_slope = theilslopes(y, x).slope
    retention_windows = {}
    for width in (25, 50, 100):
        retention_windows[str(width)] = np.median(y[-width:]) / np.median(y[:width])
    out_b = card(
        {
            "cycle_count": len(cycles),
            "last50_to_first50_median_retention": np.median(y[-50:]) / np.median(y[:50]),
            "capacity_slope_per_cycle": fit.slope,
            "linear_trend_r_squared": fit.rvalue**2,
        },
        (max(segment_slopes.values()) < 0, {"contiguous_cycle_segment_slopes": segment_slopes}),
        (robust_slope < 0, {"theil_sen_slope_per_cycle": robust_slope}),
        (min(retention_windows.values()) > 0.85 and max(retention_windows.values()) < 0.95, {"retention_window_definitions": retention_windows}),
    )
    return {"A": out_a, "B": out_b}


def information_surface(data: Path) -> dict:
    def replicate_metrics(path):
        frame = pd.read_csv(path, sep="\t")
        keys = [name for name in frame.columns if name not in {"replicate_id", "response_value"}]
        pivot = frame.pivot(index=keys, columns="replicate_id", values="response_value")
        correlations = np.log10(pivot).corr()
        pairs = [correlations.loc[left, right] for left, right in itertools.combinations(correlations.columns, 2)]
        cv = pivot.std(axis=1) / pivot.mean(axis=1)
        return {"pairwise_log_pearson": pairs, "median_cell_cv": np.median(cv)}

    replicated = replicate_metrics(data / "replicated_response_panel.tsv")
    independent = replicate_metrics(data / "independent_response_panel.tsv")
    sample = {
        "replicated_min_pairwise_log_pearson": min(replicated["pairwise_log_pearson"]),
        "independent_min_pairwise_log_pearson": min(independent["pairwise_log_pearson"]),
    }
    method = {
        "replicated_median_cell_cv": replicated["median_cell_cv"],
        "independent_median_cell_cv": independent["median_cell_cv"],
    }
    out_a = card(
        {
            "replicated_median_pairwise_log_pearson": np.median(replicated["pairwise_log_pearson"]),
            "independent_median_pairwise_log_pearson": np.median(independent["pairwise_log_pearson"]),
            "replicated_median_cell_cv": replicated["median_cell_cv"],
            "independent_median_cell_cv": independent["median_cell_cv"],
        },
        (min(sample.values()) > 0.95, sample),
        (max(method.values()) < 0.30, method),
        (min(replicated["pairwise_log_pearson"] + independent["pairwise_log_pearson"]) > 0.95, {"all_pairwise_log_correlations": {"replicated": replicated["pairwise_log_pearson"], "independent": independent["pairwise_log_pearson"]}}),
    )

    panel = pd.read_csv(data / "independent_response_panel.tsv", sep="\t")

    def invariant(frame, scale="log", baseline_bins=1):
        curves = frame.groupby(["output_channel_index", "signal_bin_index"])["response_value"].mean().unstack()
        baseline = curves.iloc[:, :baseline_bins].mean(axis=1)
        fold = curves.div(baseline, axis=0)
        transformed = np.log(fold) if scale == "log" else fold
        correlations = transformed.T.corr()
        pairs = [correlations.loc[left, right] for left, right in itertools.combinations(correlations.columns, 2)]
        return {"minimum_channel_correlation": min(pairs), "endpoint_folds": fold.iloc[:, -1].to_dict()}

    pooled = invariant(panel)
    rowwise = {
        str(index): invariant(panel[panel["input_axis_1_index"] == index])
        for index in sorted(panel["input_axis_1_index"].unique())
    }
    columnwise = {
        str(index): invariant(panel[panel["input_axis_2_index"] == index])
        for index in sorted(panel["input_axis_2_index"].unique())
    }
    raw_rowwise = {
        str(index): invariant(panel[panel["input_axis_1_index"] == index], scale="raw")
        for index in sorted(panel["input_axis_1_index"].unique())
    }
    baseline_definitions = {
        str(bin_count): invariant(panel, baseline_bins=bin_count)
        for bin_count in (1, 2, 3)
    }
    errors = pd.read_csv(data / "independent_fit_error.tsv", sep="\t")
    error_by_axis = errors.groupby("input_axis_index")["fit_error"].mean().to_dict()
    endpoint_folds = list(pooled["endpoint_folds"].values())
    minimum_rowwise = min(row["minimum_channel_correlation"] for row in rowwise.values())
    minimum_columnwise = min(row["minimum_channel_correlation"] for row in columnwise.values())
    minimum_raw_rowwise = min(row["minimum_channel_correlation"] for row in raw_rowwise.values())
    baseline_endpoint_folds = {
        definition: row["endpoint_folds"]
        for definition, row in baseline_definitions.items()
    }
    baseline_rankings = {
        definition: sorted(row["endpoint_folds"], key=row["endpoint_folds"].get, reverse=True)
        for definition, row in baseline_definitions.items()
    }
    out_b = card(
        {
            "pooled_normalized_channel_curve_correlation": pooled["minimum_channel_correlation"],
            "minimum_complete_row_channel_curve_correlation": minimum_rowwise,
            "minimum_complete_column_channel_curve_correlation": minimum_columnwise,
            "minimum_endpoint_fold_change": min(endpoint_folds),
            "maximum_endpoint_fold_change": max(endpoint_folds),
            "median_independent_fit_error": errors["fit_error"].median(),
            "maximum_independent_fit_error": errors["fit_error"].max(),
        },
        (
            minimum_rowwise > 0.90 and minimum_columnwise > 0.90,
            {"direct_complete_row_validation": rowwise, "direct_complete_column_validation": columnwise},
        ),
        (
            minimum_raw_rowwise > 0.90 and errors["fit_error"].max() < 0.16,
            {"raw_scale_complete_row_correlations": raw_rowwise, "mean_fit_error_by_input_axis": error_by_axis},
        ),
        (
            min(value for row in baseline_endpoint_folds.values() for value in row.values()) > 30
            and max(value for row in baseline_endpoint_folds.values() for value in row.values()) < 100
            and len({order[0] for order in baseline_rankings.values()}) == 1
            and len({order[-1] for order in baseline_rankings.values()}) == 1,
            {
                "pooled_endpoint_folds_under_one_two_and_three_bin_baselines": baseline_endpoint_folds,
                "channel_rankings_under_baseline_definitions": baseline_rankings,
                "interpretation": "Changing the baseline materially changes fold magnitudes; survival tests strong amplification plus stable top and bottom channels, not the algebraically invariant correlation or the interchangeable middle ranks.",
            },
        ),
    )
    return {"A": out_a, "B": out_b}


def temperature_response(data: Path) -> dict:
    def crossing_branches(x, y, minimum_points=20, allow_terminal_zero=False):
        turns = np.flatnonzero(np.sign(np.diff(x)[1:]) != np.sign(np.diff(x)[:-1])) + 1
        starts = [0, *turns.tolist()]
        ends = [*turns.tolist(), len(x) - 1]
        full_sweep_limit = 0.40 * np.max(np.abs(x))
        branches = []
        zero_tolerance = max(1e-12, np.max(np.abs(x)) * 1e-9)
        for start, end in zip(starts, ends):
            xx = x[start : end + 1]
            yy = y[start : end + 1]
            full_crossing = np.min(xx) < -full_sweep_limit and np.max(xx) > full_sweep_limit
            terminal_return = (
                allow_terminal_zero
                and abs(xx[-1]) <= zero_tolerance
                and (np.min(xx) < -full_sweep_limit or np.max(xx) > full_sweep_limit)
            )
            if end - start >= minimum_points and (full_crossing or terminal_return):
                branches.append((xx, yy))
        return branches

    def zero_value(x, y, estimator="linear", stride=1):
        order = np.argsort(x)
        ordered_x, ordered_y = x[order], y[order]
        retained = sorted(set(range(0, len(ordered_x), stride)) | {len(ordered_x) - 1, int(np.argmin(np.abs(ordered_x)))})
        xx, yy = ordered_x[retained], ordered_y[retained]
        if estimator == "linear":
            return np.interp(0, xx, yy)
        points = 10 if estimator == "local_linear" else 14
        keep = np.argsort(np.abs(xx))[:points]
        degree = 1 if estimator == "local_linear" else 2
        return np.polyfit(xx[keep], yy[keep], degree)[-1]

    def magnetic(temp, estimator="linear", stride=1, gap_window=500):
        frame = pd.read_csv(data / f"bulk_magnetization_{temp}K.tsv", sep="\t").sort_values("time_s").reset_index(drop=True)
        field = frame["field_Oe"].to_numpy()
        moment = frame["magnetic_moment_emu"].to_numpy()
        branches = crossing_branches(field, moment)
        zeros = [zero_value(x, y, estimator=estimator, stride=stride) for x, y in branches]
        grid = np.linspace(-gap_window, gap_window, 101)
        branch_curves = []
        for x, y in branches:
            order = np.argsort(x)
            branch_curves.append(np.interp(grid, x[order], y[order]))
        if len(branch_curves) != 2:
            raise RuntimeError(f"Expected two complete cross-zero branches at {temp} K, found {len(branch_curves)}")
        return {
            "zero_field_branch_separation_emu": max(zeros) - min(zeros),
            "median_absolute_branch_gap_emu": np.median(np.abs(branch_curves[0] - branch_curves[1])),
            "complete_cross_zero_branch_count": len(branches),
            "branch_zero_values": zeros,
        }

    base_magnetic = {str(temp): magnetic(temp) for temp in (300, 860, 870)}
    stride_checks = {
        str(stride): {str(temp): magnetic(temp, stride=stride) for temp in (300, 860, 870)}
        for stride in (1, 2, 3, 4)
    }
    estimator_checks = {
        estimator: {str(temp): magnetic(temp, estimator=estimator) for temp in (300, 860, 870)}
        for estimator in ("linear", "local_linear", "local_quadratic")
    }
    window_checks = {
        str(window): {str(temp): magnetic(temp, gap_window=window) for temp in (300, 860, 870)}
        for window in (250, 500, 1000)
    }
    low_field_ratio = base_magnetic["300"]["zero_field_branch_separation_emu"] / base_magnetic["870"]["zero_field_branch_separation_emu"]
    out_a = card(
        {
            "zero_field_branch_separation_300K_emu": base_magnetic["300"]["zero_field_branch_separation_emu"],
            "zero_field_branch_separation_860K_emu": base_magnetic["860"]["zero_field_branch_separation_emu"],
            "zero_field_branch_separation_870K_emu": base_magnetic["870"]["zero_field_branch_separation_emu"],
            "branch_separation_ratio_300K_to_870K": low_field_ratio,
        },
        (
            all(rows["300"]["zero_field_branch_separation_emu"] > rows["860"]["zero_field_branch_separation_emu"] > rows["870"]["zero_field_branch_separation_emu"] for rows in stride_checks.values()),
            {"branch_preserving_systematic_thinning": stride_checks},
        ),
        (
            all(rows["300"]["zero_field_branch_separation_emu"] > rows["870"]["zero_field_branch_separation_emu"] * 10 for rows in estimator_checks.values()),
            {"zero_crossing_estimators": estimator_checks},
        ),
        (
            all(rows["300"]["median_absolute_branch_gap_emu"] > rows["870"]["median_absolute_branch_gap_emu"] * 5 for rows in window_checks.values()),
            {"low_field_branch_gap_windows": window_checks},
        ),
    )

    def electric(path, value_column):
        frame = pd.read_csv(data / path, sep="\t")
        x = frame["sequence_coordinate"].to_numpy()
        y = frame[value_column].to_numpy()
        branches = crossing_branches(x, y, allow_terminal_zero=True)
        zeros = [zero_value(xx, yy) for xx, yy in branches]
        stride_separations = {}
        for stride in (1, 2, 3, 4):
            values = [zero_value(xx, yy, stride=stride) for xx, yy in branches]
            stride_separations[str(stride)] = max(values) - min(values)
        trend = linregress(x, y)
        detrended = y - (trend.intercept + trend.slope * x)
        return {
            "branch_zero_separation": max(zeros) - min(zeros),
            "signed_loop_area": np.trapezoid(y, x),
            "detrended_signed_loop_area": np.trapezoid(detrended, x),
            "branch_zero_values": zeros,
            "complete_cross_zero_branch_count": len(branches),
            "stride_separations": stride_separations,
        }

    raw = electric("electric_current_voltage.tsv", "current_or_voltage_signal")
    corrected = electric("electric_switching_corrected.tsv", "polarization_like_signal")
    out_b = card(
        {
            "corrected_branch_zero_separation": corrected["branch_zero_separation"],
            "corrected_absolute_loop_area": abs(corrected["signed_loop_area"]),
            "raw_branch_zero_separation": raw["branch_zero_separation"],
            "raw_absolute_loop_area": abs(raw["signed_loop_area"]),
        },
        (
            raw["complete_cross_zero_branch_count"] == 2
            and corrected["complete_cross_zero_branch_count"] == 2
            and min(raw["stride_separations"].values()) > 10
            and min(corrected["stride_separations"].values()) > 0.30,
            {"strict_cross_zero_branches_and_systematic_thinning": {"raw": raw, "corrected": corrected}},
        ),
        (abs(corrected["detrended_signed_loop_area"]) > 8 and abs(raw["detrended_signed_loop_area"]) > 30_000, {"linear_detrended_loop_areas": {"raw": raw["detrended_signed_loop_area"], "corrected": corrected["detrended_signed_loop_area"]}}),
        (corrected["branch_zero_separation"] > 0.30 and raw["branch_zero_separation"] > 10, {"direction_aware_branch_separations": {"raw": raw["branch_zero_separation"], "corrected": corrected["branch_zero_separation"]}}),
    )
    return {"A": out_a, "B": out_b}


def membrane(data: Path) -> dict:
    bias = pd.read_csv(data / "bias_current_curves.tsv", sep="\t")
    slopes = {}
    for name, group in bias.groupby("sample_class"):
        x = group["bias_V"].to_numpy()
        y = group["current_A"].to_numpy()
        slopes[name] = {
            "ols": linregress(x, y).slope,
            "through_origin": np.sum(x * y) / np.sum(x**2),
            "theil_sen": theilslopes(y, x).slope,
        }
    slope_ratios = {
        method: slopes["modified_membrane"][method] / slopes["reference_membrane"][method]
        for method in slopes["modified_membrane"]
    }
    conductivity = pd.read_csv(data / "conductivity_replicates.tsv", sep="\t")
    conductivity_medians = conductivity.groupby("sample_class")["proton_conductivity_mS_cm2"].median().to_dict()
    conductivity_means = conductivity.groupby("sample_class")["proton_conductivity_mS_cm2"].mean().to_dict()
    conductivity_ratio = conductivity_medians["modified_membrane"] / conductivity_medians["reference_membrane"]
    leave_one_specimen_ratios = {}
    for index, row in conductivity.iterrows():
        retained = conductivity.drop(index)
        medians = retained.groupby("sample_class")["proton_conductivity_mS_cm2"].median()
        leave_one_specimen_ratios[row["replicate_id"]] = medians["modified_membrane"] / medians["reference_membrane"]
    local = pd.read_csv(data / "local_current_distributions.tsv", sep="\t")
    local_medians = local.groupby("sample_class")["current_pA"].median().to_dict()
    local_ratio = local_medians["modified_membrane"] / local_medians["reference_membrane"]
    local_descriptive_ratios = {}
    for trim in (0.0, 0.05, 0.10, 0.20):
        summaries = {}
        for name, group in local.groupby("sample_class"):
            low, high = group["current_pA"].quantile([trim, 1 - trim]) if trim else (group["current_pA"].min(), group["current_pA"].max())
            summaries[name] = group.loc[group["current_pA"].between(low, high), "current_pA"].median()
        local_descriptive_ratios[str(trim)] = summaries["modified_membrane"] / summaries["reference_membrane"]
    out_a = card(
        {
            "modified_bias_conductance_A_per_V": slopes["modified_membrane"]["ols"],
            "reference_bias_conductance_A_per_V": slopes["reference_membrane"]["ols"],
            "bias_conductance_ratio": slope_ratios["ols"],
            "modified_conductivity_median_mS_cm2": conductivity_medians["modified_membrane"],
            "reference_conductivity_median_mS_cm2": conductivity_medians["reference_membrane"],
            "conductivity_median_ratio": conductivity_ratio,
            "descriptive_local_current_median_ratio": local_ratio,
        },
        (min(leave_one_specimen_ratios.values()) > 20, {"leave_one_conductivity_specimen_ratios": leave_one_specimen_ratios}),
        (
            min(slope_ratios.values()) > 10 and conductivity_means["modified_membrane"] / conductivity_means["reference_membrane"] > 20,
            {"bias_slope_ratios_by_method": slope_ratios, "conductivity_class_means": conductivity_means},
        ),
        (
            min(local_descriptive_ratios.values()) > 20,
            {"unparented_local_vector_descriptive_trim_sensitivity": local_descriptive_ratios, "inferential_claim": False},
        ),
    )

    spatial = pd.read_csv(data / "spatial_current_map.tsv", sep="\t")

    def blocks(size):
        frame = spatial.assign(
            block_x=np.floor(spatial["x_um"] / size).astype(int),
            block_y=np.floor(spatial["y_um"] / size).astype(int),
        )
        medians = frame.groupby(["block_x", "block_y"])["current_pA"].median()
        return {"block_count": len(medians), "minimum_median_pA": medians.min(), "maximum_median_pA": medians.max(), "fraction_median_above_10_pA": np.mean(medians > 10)}

    block_metrics = {str(size): blocks(size) for size in (0.5, 0.7, 1.0, 1.4)}
    thresholds = {str(value): np.mean(spatial["current_pA"] > value) for value in (1, 5, 10, 50, 100)}
    quadrant_leave = []
    for high_x in (False, True):
        for high_y in (False, True):
            excluded = ((spatial["x_um"] >= 1.4) == high_x) & ((spatial["y_um"] >= 1.4) == high_y)
            quadrant_leave.append(np.mean(spatial.loc[~excluded, "current_pA"] > 10))
    base_blocks = block_metrics["1.0"]
    out_b = card(
        {
            "spatial_pixel_count": len(spatial),
            "spatial_median_current_pA": spatial["current_pA"].median(),
            "fraction_pixels_above_10_pA": np.mean(spatial["current_pA"] > 10),
            "one_um_block_minimum_median_pA": base_blocks["minimum_median_pA"],
            "one_um_block_maximum_median_pA": base_blocks["maximum_median_pA"],
        },
        (min(quadrant_leave) > 0.40 and max(quadrant_leave) < 0.60, {"leave_one_spatial_quadrant_hot_fraction": quadrant_leave}),
        (all(row["maximum_median_pA"] > 100 and row["minimum_median_pA"] < 0.1 for row in block_metrics.values()), {"contiguous_block_summaries": block_metrics}),
        (thresholds["1"] > 0.50 and thresholds["100"] > 0.30, {"hot_pixel_threshold_sensitivity": thresholds}),
    )
    return {"A": out_a, "B": out_b}


def governing_equations(data: Path) -> dict:
    archive = np.load(data / "spatiotemporal_fields.npz")
    coordinate_metadata = json.loads((data / "coordinate_definitions.json").read_text(encoding="utf-8"))
    time = archive["synthetic_time"]
    space = archive["synthetic_space"]
    field = archive["synthetic_field"]

    term_names = [
        "u", "u_squared", "u_cubed", "u_x", "u_times_u_x", "u_squared_times_u_x",
        "u_xx", "u_times_u_xx", "sin_x_sin_t", "cos_x_sin_t", "sin_x_cos_t",
        "cos_x_cos_t", "intercept",
    ]

    def pde_library(values, boundary=2):
        dt = time[1] - time[0]
        dx = space[1] - space[0]
        ut = (values[2:, boundary:-boundary] - values[:-2, boundary:-boundary]) / (2 * dt)
        ux_all = (values[1:-1, 2:] - values[1:-1, :-2]) / (2 * dx)
        uxx_all = (values[1:-1, 2:] - 2 * values[1:-1, 1:-1] + values[1:-1, :-2]) / dx**2
        crop = boundary - 1
        stop = -crop if crop else None
        ux = ux_all[:, crop:stop]
        uxx = uxx_all[:, crop:stop]
        interior = values[1:-1, boundary:-boundary]
        grid_t, grid_x = np.meshgrid(time[1:-1], space[boundary:-boundary], indexing="ij")
        source = np.sin(grid_x) * np.sin(grid_t)
        design = np.stack([
            interior,
            interior**2,
            interior**3,
            ux,
            interior * ux,
            interior**2 * ux,
            uxx,
            interior * uxx,
            source,
            np.cos(grid_x) * np.sin(grid_t),
            np.sin(grid_x) * np.cos(grid_t),
            np.cos(grid_x) * np.cos(grid_t),
            np.ones_like(interior),
        ], axis=-1)
        return design, ut, grid_t

    def sparse_fit(values, boundary=2, threshold=0.02, train_mask=None, evaluation_mask=None):
        design, target, grid_t = pde_library(values, boundary=boundary)
        if train_mask is None:
            train_mask = np.ones(design.shape[:2], dtype=bool)
        elif train_mask.ndim == 1:
            train_mask = np.broadcast_to(train_mask[:, None], design.shape[:2])
        train_design = design[train_mask]
        train_target = target[train_mask]
        active = np.ones(len(term_names), dtype=bool)
        coefficients = np.zeros(len(term_names))
        for _ in range(len(term_names) + 1):
            coefficients[:] = 0
            coefficients[active] = np.linalg.lstsq(train_design[:, active], train_target, rcond=None)[0]
            updated = np.abs(coefficients) >= threshold
            if np.array_equal(updated, active):
                break
            active = updated
        if evaluation_mask is None:
            evaluation_mask = train_mask
        elif evaluation_mask.ndim == 1:
            evaluation_mask = np.broadcast_to(evaluation_mask[:, None], design.shape[:2])
        evaluation_design = design[evaluation_mask]
        evaluation_target = target[evaluation_mask]
        prediction = evaluation_design @ coefficients
        r_squared = 1 - np.sum((evaluation_target - prediction) ** 2) / np.sum((evaluation_target - np.mean(evaluation_target)) ** 2)
        coefficient_map = dict(zip(term_names, coefficients))
        return {
            "candidate_library_size": len(term_names),
            "selected_term_count": int(np.sum(active)),
            "selected_terms": [term_names[index] for index in np.flatnonzero(active)],
            "coefficient_u_times_u_x": coefficient_map["u_times_u_x"],
            "coefficient_u_xx": coefficient_map["u_xx"],
            "coefficient_sin_x_sin_t": coefficient_map["sin_x_sin_t"],
            "r_squared": r_squared,
        }

    expected_terms = {"u_times_u_x", "u_xx", "sin_x_sin_t"}
    base = sparse_fit(field)
    grid_time = time[1:-1]
    holdout_windows = {"early": (0.1, 3.3), "middle": (3.4, 6.6), "late": (6.7, 9.9)}
    held_blocks = {}
    for name, (low, high) in holdout_windows.items():
        held = (grid_time >= low) & (grid_time <= high)
        held_blocks[name] = sparse_fit(field, train_mask=~held, evaluation_mask=held)
    smoothed = {str(sigma): sparse_fit(gaussian_filter(field, (sigma, sigma))) for sigma in (0.25, 0.5)}
    smoothing_stress_boundary = sparse_fit(gaussian_filter(field, (1.0, 1.0)))
    threshold_checks = {str(value): sparse_fit(field, threshold=value) for value in (0.005, 0.01, 0.02, 0.05, 0.10)}
    boundary_checks = {str(width): sparse_fit(field, boundary=width) for width in (2, 3, 4, 5, 10)}
    out_a = card(
        base,
        (
            all(set(row["selected_terms"]) == expected_terms and row["r_squared"] > 0.98 for row in held_blocks.values()),
            {"train_on_two_time_blocks_predict_held_block": held_blocks},
        ),
        (
            all(set(row["selected_terms"]) == expected_terms and row["r_squared"] > 0.98 for row in smoothed.values()),
            {"admitted_gaussian_smoothing": smoothed, "sigma_1_stress_boundary_not_admitted": smoothing_stress_boundary},
        ),
        (
            all(set(row["selected_terms"]) == expected_terms and row["r_squared"] > 0.98 for row in [*threshold_checks.values(), *boundary_checks.values()]),
            {"selection_thresholds": threshold_checks, "spatial_boundary_exclusions": boundary_checks},
        ),
    )

    experimental_time = archive["experimental_time"]
    experimental_space = archive["experimental_space"]
    experimental = archive["experimental_field"]
    totals = np.trapezoid(experimental, experimental_space, axis=1)
    ratios = totals[-1] / totals[0]
    channel_correlations = [
        np.corrcoef(experimental[:, :, left].ravel(), experimental[:, :, right].ravel())[0, 1]
        for left, right in itertools.combinations(range(experimental.shape[2]), 2)
    ]
    spatial_definitions = {}
    for fraction in (0.0, 0.1, 0.2, 0.25):
        trim = int(len(experimental_space) * fraction)
        stop = len(experimental_space) - trim
        values = np.trapezoid(experimental[:, trim:stop, :], experimental_space[trim:stop], axis=1)
        spatial_definitions[str(fraction)] = {
            "last_to_first_ratios": (values[-1] / values[0]).tolist(),
            "all_channels_strictly_increasing": bool(np.all(np.diff(values, axis=0) > 0)),
        }
    summed = np.sum(experimental, axis=1)
    summed_ratios = summed[-1] / summed[0]
    leave_channel = {}
    for index in range(experimental.shape[2]):
        keep = [value for value in range(experimental.shape[2]) if value != index]
        leave_channel[str(index)] = float(np.min(ratios[keep]))
    out_b = card(
        {
            "experimental_time_count": len(experimental_time),
            "channel_count": experimental.shape[2],
            "experimental_time_unit": coordinate_metadata["experimental_field"]["time_coordinate"],
            "experimental_field_semantics": coordinate_metadata["experimental_field"]["field"],
            "minimum_last_to_first_integrated_ratio": np.min(ratios),
            "maximum_last_to_first_integrated_ratio": np.max(ratios),
            "minimum_cross_channel_field_correlation": min(channel_correlations),
        },
        (min(leave_channel.values()) > 3, {"leave_one_channel_minimum_growth_ratio": leave_channel}),
        (np.min(summed_ratios) > 3, {"sum_based_last_to_first_ratios": summed_ratios.tolist()}),
        (all(row["all_channels_strictly_increasing"] and min(row["last_to_first_ratios"]) > 3 for row in spatial_definitions.values()), {"spatial_domain_definitions": spatial_definitions}),
    )
    return {"A": out_a, "B": out_b}


HANDLERS = {
    "Astronomy_04_gravity_spectral_structure": astronomy,
    "Chemistry_07_photoionization_delay": chemistry,
    "EarthScience_03_ice_particle_composition": earth_particles,
    "EarthScience_04_ice_core_event_structure": earth_events,
    "Energy_01_photocatalytic_tradeoffs": photocatalysis,
    "Energy_02_flow_battery_operating_boundary": flow_battery,
    "Information_03_signal_surface_invariants": information_surface,
    "Material_02_temperature_response_boundaries": temperature_response,
    "Material_03_membrane_selective_permeation": membrane,
    "Math_01_governing_equation_discovery": governing_equations,
}

# Kept in a separate module so the established ten-task actions remain easy to
# audit. The public CLI and generated wrappers still resolve one unified map.
from scientific_actions_multimodal import MULTIMODAL_HANDLERS

HANDLERS.update(MULTIMODAL_HANDLERS)


def run(task_id: str, data: Path, card_id: str | None = None, family: str = "all") -> dict:
    if task_id not in HANDLERS:
        raise ValueError(f"Unknown task: {task_id}")
    result = HANDLERS[task_id](data)
    if card_id is None:
        return native(result)
    selected = result[card_id]
    if family == "recompute":
        return {"task_id": task_id, "card_id": card_id, "values": selected["values"]}
    if family in selected["perturbations"]:
        return {"task_id": task_id, "card_id": card_id, "family": family, **selected["perturbations"][family]}
    return {"task_id": task_id, "card_id": card_id, **selected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(HANDLERS))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--card", choices=("A", "B"))
    parser.add_argument("--family", choices=("all", "recompute", "sample", "method", "definition"), default="all")
    args = parser.parse_args()
    print(json.dumps(run(args.task, args.data, args.card, args.family), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
