"""Executable public-data-only Actions for multimodal tasks.

The functions deliberately return two non-exhaustive anchors, not answer keys.
Every perturbation operates on a real analysis unit, method, or definition and
records the changed numerical result used by its survival rule.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, theilslopes

from scientific_actions_tabular import card


def _rho(x, y) -> float:
    return float(spearmanr(np.asarray(x), np.asarray(y), nan_policy="omit").statistic)


def _pearson(x, y) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    return float(np.corrcoef(frame["x"], frame["y"])[0, 1])


def stellar_spectra(data: Path) -> dict:
    grid = pd.read_csv(data / "photon_flux_grid.tsv", sep="\t")

    def shortwave(limit: float) -> dict:
        rows = []
        for (temperature, wavelength), group in grid[grid["wavelength_nm"] <= limit].groupby(
            ["effective_temperature_K", "wavelength_nm"]
        ):
            ordered = group.sort_values("metallicity_index")
            rows.append(
                {
                    "temperature": temperature,
                    "wavelength": wavelength,
                    "rho": _rho(ordered["metallicity_index"], ordered["photon_flux_cm2"]),
                    "endpoint_ratio": ordered["photon_flux_cm2"].iloc[-1] / ordered["photon_flux_cm2"].iloc[0],
                }
            )
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

    base = shortwave(20_000)
    cutoffs = {str(limit): shortwave(limit) for limit in (20_000, 22_000, 25_000)}
    temperature_holdouts = {
        str(temp): shortwave(20_000)["by_temperature"][temp]
        for temp in sorted(grid["effective_temperature_K"].unique())
    }
    out_a = card(
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

    response = pd.read_csv(data / "normalized_spectral_response.tsv", sep="\t")
    levels = list(response.columns[1:])
    peaks = {name: float(response.loc[response[name].idxmax(), "wavelength_nm"]) for name in levels}
    pearson = response[levels].corr()
    spearman = response[levels].corr(method="spearman")
    pair_pearson = [pearson.loc[a, b] for a, b in itertools.combinations(levels, 2)]
    pair_spearman = [spearman.loc[a, b] for a, b in itertools.combinations(levels, 2)]
    leave_level = {
        name: float(max(value for key, value in peaks.items() if key != name) - min(value for key, value in peaks.items() if key != name))
        for name in levels
    }
    centroid = {
        name: float(np.average(response["wavelength_nm"], weights=np.maximum(response[name], 0)))
        for name in levels
    }
    out_b = card(
        {
            "response_level_count": len(levels),
            "minimum_pairwise_spectral_pearson": min(pair_pearson),
            "spectral_peak_wavelength_min_nm": min(peaks.values()),
            "spectral_peak_wavelength_max_nm": max(peaks.values()),
            "spectral_peak_span_nm": max(peaks.values()) - min(peaks.values()),
        },
        (max(leave_level.values()) <= 20, {"leave_one_response_level_peak_span_nm": leave_level}),
        (min(pair_spearman) > 0.90, {"pairwise_spectral_spearman": pair_spearman}),
        (
            max(centroid.values()) - min(centroid.values()) < 80,
            {"intensity_weighted_centroid_wavelength_nm": centroid},
        ),
    )
    return {"A": out_a, "B": out_b}


def catalysis_landscape(data: Path) -> dict:
    panel1 = pd.read_csv(data / "observation_set_1.tsv", sep="\t")

    def categorical_span(frame: pd.DataFrame, outcome: str, reducer: str = "median") -> float:
        grouped = frame.groupby("numeric_feature_4")[outcome]
        values = grouped.median() if reducer == "median" else grouped.mean()
        return float(values.max() - values.min())

    split_spans = {
        "even_observations": categorical_span(panel1[panel1["observation_index"] % 2 == 0], "numeric_feature_7"),
        "odd_observations": categorical_span(panel1[panel1["observation_index"] % 2 == 1], "numeric_feature_7"),
        "first_half": categorical_span(panel1.iloc[: len(panel1) // 2], "numeric_feature_7"),
        "second_half": categorical_span(panel1.iloc[len(panel1) // 2 :], "numeric_feature_7"),
    }
    mean_span = categorical_span(panel1, "numeric_feature_7", reducer="mean")
    alternate_outcome_span = categorical_span(panel1, "numeric_feature_6")
    out_a = card(
        {
            "observation_count": len(panel1),
            "category_count_feature_4": panel1["numeric_feature_4"].nunique(),
            "feature_7_category_median_span": categorical_span(panel1, "numeric_feature_7"),
            "feature_6_category_median_span": alternate_outcome_span,
        },
        (min(split_spans.values()) > 0.80, {"disjoint_observation_split_spans": split_spans}),
        (mean_span > 0.80, {"category_mean_span": mean_span}),
        (alternate_outcome_span > 0.70, {"alternate_response_feature_median_span": alternate_outcome_span}),
    )

    panel2 = pd.read_csv(data / "observation_set_2.tsv", sep="\t").dropna(subset=["numeric_feature_1", "numeric_feature_2"])
    panel3 = pd.read_csv(data / "observation_set_3.tsv", sep="\t")
    panel4 = pd.read_csv(data / "observation_set_4.tsv", sep="\t")
    mappings = {
        "panel_2_features_1_2": (panel2, "numeric_feature_1", "numeric_feature_2"),
        "panel_3_features_3_4": (panel3, "numeric_feature_3", "numeric_feature_4"),
        "panel_4_features_4_5": (panel4, "numeric_feature_4", "numeric_feature_5"),
    }
    pearson = {name: _pearson(frame[x], frame[y]) for name, (frame, x, y) in mappings.items()}
    rank = {name: _rho(frame[x], frame[y]) for name, (frame, x, y) in mappings.items()}
    halves = {}
    for name, (frame, x, y) in mappings.items():
        halves[name] = [
            _pearson(part[x], part[y])
            for part in (frame.iloc[: len(frame) // 2], frame.iloc[len(frame) // 2 :])
        ]
    trimmed = {
        name: _pearson(frame.iloc[3:-3][x], frame.iloc[3:-3][y])
        for name, (frame, x, y) in mappings.items()
    }
    out_b = card(
        {
            "panel_2_mapping_pearson": pearson["panel_2_features_1_2"],
            "panel_3_mapping_pearson": pearson["panel_3_features_3_4"],
            "panel_4_mapping_pearson": pearson["panel_4_features_4_5"],
            "minimum_selected_cross_panel_mapping_pearson": min(pearson.values()),
        },
        (min(value for rows in halves.values() for value in rows) > 0.70, {"contiguous_half_correlations": halves}),
        (min(rank.values()) > 0.80, {"selected_mapping_spearman": rank}),
        (min(trimmed.values()) > 0.75, {"three_endpoint_trimmed_correlations": trimmed}),
    )
    return {"A": out_a, "B": out_b}


def ocean_record(data: Path) -> dict:
    frame = pd.read_csv(data / "observations.tsv", sep="\t")
    depth, age1, age2, proxy = frame.columns
    ages = frame[[age1, age2]].dropna()
    blocks = {}
    for index, rows in enumerate(np.array_split(np.arange(len(ages)), 5)):
        block = ages.iloc[rows]
        blocks[str(index)] = {
            "spearman": _rho(block[age1], block[age2]),
            "median_absolute_gap": float(np.median(np.abs(block[age1] - block[age2]))),
        }
    gaps = np.abs(ages[age1] - ages[age2])
    out_a = card(
        {
            "complete_row_count": len(frame),
            "dual_age_spearman": _rho(ages[age1], ages[age2]),
            "dual_age_pearson": _pearson(ages[age1], ages[age2]),
            "median_absolute_age_gap_ka": np.median(gaps),
            "p95_absolute_age_gap_ka": np.percentile(gaps, 95),
        },
        (min(row["spearman"] for row in blocks.values()) > 0.999, {"five_contiguous_age_blocks": blocks}),
        (_pearson(ages[age1], ages[age2]) > 0.999, {"pearson_coordinate_agreement": _pearson(ages[age1], ages[age2])}),
        (
            float(np.mean(gaps <= 0.25)) > 0.90,
            {"fraction_age_coordinate_gaps_at_most_0_25_ka": float(np.mean(gaps <= 0.25))},
        ),
    )

    observed = frame[[age1, proxy]].dropna().sort_values(age1).reset_index(drop=True)

    def proxy_blocks(count: int, reducer: str = "median") -> dict:
        rows = np.array_split(np.arange(len(observed)), count)
        values = []
        for indices in rows:
            series = observed.iloc[indices][proxy]
            values.append(float(series.median() if reducer == "median" else series.mean()))
        return {"values": values, "range": max(values) - min(values), "largest_adjacent_jump": max(abs(np.diff(values)))}

    base = proxy_blocks(8)
    leave_block = {}
    block_index = np.array_split(np.arange(len(observed)), 8)
    for index, omitted in enumerate(block_index):
        retained = observed.drop(index=omitted).reset_index(drop=True)
        values = [
            float(retained.iloc[indices][proxy].median())
            for indices in np.array_split(np.arange(len(retained)), 7)
        ]
        leave_block[str(index)] = max(values) - min(values)
    definitions = {str(count): proxy_blocks(count) for count in (6, 8, 10)}
    mean_blocks = proxy_blocks(8, reducer="mean")
    out_b = card(
        {
            "global_proxy_age_spearman": _rho(observed[age1], observed[proxy]),
            "eight_block_proxy_median_range": base["range"],
            "eight_block_largest_adjacent_median_jump": base["largest_adjacent_jump"],
            "proxy_nonmissing_count": len(observed),
        },
        (min(leave_block.values()) > 0.35, {"leave_one_contiguous_block_ranges": leave_block}),
        (mean_blocks["range"] > 0.30, {"eight_block_mean_summary": mean_blocks}),
        (min(row["range"] for row in definitions.values()) > 0.45, {"block_count_definitions": definitions}),
    )
    return {"A": out_a, "B": out_b}


def photoreduction(data: Path) -> dict:
    frame = pd.read_csv(data / "photoproduct_timecourses.tsv", sep="\t")
    bounded = frame.dropna(subset=["secondary_component_level"])
    products = ["hydrogen_yield_umol_g", "carbon_monoxide_yield_umol_g"]

    def peak_levels(rows: pd.DataFrame, mode: str = "endpoint") -> dict:
        if mode == "endpoint":
            values = rows.sort_values("illumination_duration_h").groupby("formulation_id").tail(1)
        elif mode == "integral":
            values = rows.groupby(["formulation_id", "secondary_component_level"], as_index=False)[products].sum()
        else:
            values = rows.groupby(["formulation_id", "secondary_component_level"], as_index=False)[products].mean()
        return {product: float(values.loc[values[product].idxmax(), "secondary_component_level"]) for product in products}

    endpoints = bounded.sort_values("illumination_duration_h").groupby("formulation_id").tail(1)
    endpoint_peaks = peak_levels(bounded)
    time_peaks = {
        str(time): {
            product: float(group.loc[group[product].idxmax(), "secondary_component_level"])
            for product in products
        }
        for time, group in bounded.groupby("illumination_duration_h")
    }
    endpoint_folds = {}
    for product in products:
        edge = endpoints[endpoints["secondary_component_level"].isin([0.0, 4.0])][product].max()
        endpoint_folds[product] = float(endpoints[product].max() / edge)
    out_a = card(
        {
            "hydrogen_peak_secondary_component_level": endpoint_peaks[products[0]],
            "carbon_monoxide_peak_secondary_component_level": endpoint_peaks[products[1]],
            "hydrogen_peak_to_edge_fold": endpoint_folds[products[0]],
            "carbon_monoxide_peak_to_edge_fold": endpoint_folds[products[1]],
        },
        (
            all(all(value == 2.0 for value in row.values()) for row in time_peaks.values()),
            {"complete_time_slice_peak_levels": time_peaks},
        ),
        (
            all(value == 2.0 for value in peak_levels(bounded, mode="integral").values()),
            {"time_integrated_peak_levels": peak_levels(bounded, mode="integral")},
        ),
        (
            all(value == 2.0 for value in peak_levels(bounded, mode="mean").values()),
            {"time_mean_peak_levels": peak_levels(bounded, mode="mean")},
        ),
    )

    by_time = {
        str(time): {
            "pearson": _pearson(group[products[0]], group[products[1]]),
            "spearman": _rho(group[products[0]], group[products[1]]),
        }
        for time, group in frame.groupby("illumination_duration_h")
    }
    slopes = {
        formulation: {
            product: float(linregress(group["illumination_duration_h"], group[product]).slope)
            for product in products
        }
        for formulation, group in frame.groupby("formulation_id")
    }
    out_b = card(
        {
            "all_observation_product_pearson": _pearson(frame[products[0]], frame[products[1]]),
            "all_observation_product_spearman": _rho(frame[products[0]], frame[products[1]]),
            "formulation_count": frame["formulation_id"].nunique(),
            "all_formulation_product_slopes_positive": all(value > 0 for row in slopes.values() for value in row.values()),
        },
        (min(row["spearman"] for row in by_time.values()) > 0.65, {"complete_time_slice_product_correlations": by_time}),
        (_pearson(frame[products[0]], frame[products[1]]) > 0.75, {"raw_scale_pearson": _pearson(frame[products[0]], frame[products[1]])}),
        (all(value > 0 for row in slopes.values() for value in row.values()), {"within_formulation_time_slopes": slopes}),
    )
    return {"A": out_a, "B": out_b}


def interface_response(data: Path) -> dict:
    main = pd.read_csv(data / "anonymous_condition_response_curves.tsv", sep="\t")
    independent = pd.read_csv(data / "independent_condition_check_curves.tsv", sep="\t")

    def curve_metrics(frame: pd.DataFrame) -> dict:
        rows = {}
        for condition, group in frame.groupby("condition_id"):
            group = group.sort_values("temperature_like_setting")
            first = float(group["rate_like_response"].iloc[0])
            last = float(group["rate_like_response"].iloc[-1])
            rows[condition] = {
                "spearman": _rho(group["temperature_like_setting"], group["rate_like_response"]),
                "endpoint_ratio_with_offset": (last + 1e-6) / (first + 1e-6),
                "log_linear_slope": float(
                    linregress(group["temperature_like_setting"], np.log(group["rate_like_response"].clip(lower=1e-6))).slope
                ),
            }
        return rows

    main_metrics = curve_metrics(main)
    trimmed_main = pd.concat(
        [group.sort_values("temperature_like_setting").iloc[1:-1] for _, group in main.groupby("condition_id")],
        ignore_index=True,
    )
    trimmed_metrics = curve_metrics(trimmed_main)
    halves = {}
    for condition, group in main.groupby("condition_id"):
        group = group.sort_values("temperature_like_setting")
        halves[condition] = {
            "lower_setting_slope": float(linregress(group.iloc[:6]["temperature_like_setting"], group.iloc[:6]["rate_like_response"]).slope),
            "upper_setting_slope": float(linregress(group.iloc[5:]["temperature_like_setting"], group.iloc[5:]["rate_like_response"]).slope),
        }
    out_a = card(
        {
            "main_condition_count": len(main_metrics),
            "minimum_condition_temperature_spearman": min(row["spearman"] for row in main_metrics.values()),
            "median_endpoint_growth_ratio_with_offset": np.median([row["endpoint_ratio_with_offset"] for row in main_metrics.values()]),
            "minimum_log_linear_slope": min(row["log_linear_slope"] for row in main_metrics.values()),
        },
        (min(row["spearman"] for row in trimmed_metrics.values()) > 0.98, {"endpoint_trimmed_complete_curves": trimmed_metrics}),
        (min(row["log_linear_slope"] for row in main_metrics.values()) > 0, {"condition_log_linear_slopes": main_metrics}),
        (min(value for row in halves.values() for value in row.values()) > 0, {"lower_and_upper_setting_slopes": halves}),
    )

    independent_metrics = curve_metrics(independent)
    shared = sorted(set(main["condition_id"]) & set(independent["condition_id"]))
    shared_correlations = {}
    shared_normalized_errors = {}
    for condition in shared:
        left = main[main["condition_id"] == condition].sort_values("temperature_like_setting")
        right = independent[independent["condition_id"] == condition].sort_values("temperature_like_setting")
        merged = left.merge(right, on="temperature_like_setting", suffixes=("_main", "_check"))
        shared_correlations[condition] = _pearson(merged["rate_like_response_main"], merged["rate_like_response_check"])
        left_norm = merged["rate_like_response_main"] / merged["rate_like_response_main"].max()
        right_norm = merged["rate_like_response_check"] / merged["rate_like_response_check"].max()
        shared_normalized_errors[condition] = float(np.median(np.abs(left_norm - right_norm)))
    leave_setting = {}
    for position in range(1, 10):
        retained = pd.concat(
            [
                group.sort_values("temperature_like_setting").drop(
                    group.sort_values("temperature_like_setting").index[position]
                )
                for _, group in independent.groupby("condition_id")
            ],
            ignore_index=True,
        )
        leave_setting[str(position)] = min(row["spearman"] for row in curve_metrics(retained).values())
    out_b = card(
        {
            "independent_condition_count": len(independent_metrics),
            "minimum_independent_curve_spearman": min(row["spearman"] for row in independent_metrics.values()),
            "shared_condition_count": len(shared),
            "minimum_shared_curve_pearson": min(shared_correlations.values()),
        },
        (min(leave_setting.values()) > 0.98, {"leave_one_interior_setting_minimum_spearman": leave_setting}),
        (max(shared_normalized_errors.values()) < 0.20, {"shared_curve_normalized_median_absolute_errors": shared_normalized_errors}),
        (min(row["log_linear_slope"] for row in independent_metrics.values()) > 0, {"independent_log_linear_slopes": independent_metrics}),
    )
    return {"A": out_a, "B": out_b}


def biohybrid(data: Path) -> dict:
    frame = pd.read_csv(data / "perturbation_time_response.tsv", sep="\t").sort_values("time")
    series = ["control_mean", "condition_a_mean", "condition_b_mean"]
    slopes = {name: float(linregress(frame["time"], frame[name]).slope) for name in series}
    endpoint_ratios = {name: float(frame[name].iloc[-1] / frame[name].iloc[0]) for name in series}
    leave_interior = {}
    for index in range(1, len(frame) - 1):
        retained = frame.drop(frame.index[index])
        leave_interior[str(index)] = {name: float(linregress(retained["time"], retained[name]).slope) for name in series}
    rank_slopes = {name: _rho(frame["time"], frame[name]) for name in series}
    decline_from_peak = {
        name: float((frame[name].max() - frame[name].iloc[-1]) / frame[name].max())
        for name in series
    }
    out_a = card(
        {
            "control_endpoint_ratio": endpoint_ratios["control_mean"],
            "condition_a_endpoint_ratio": endpoint_ratios["condition_a_mean"],
            "condition_b_endpoint_ratio": endpoint_ratios["condition_b_mean"],
            "condition_a_linear_slope": slopes["condition_a_mean"],
            "condition_b_linear_slope": slopes["condition_b_mean"],
        },
        (
            all(abs(row["control_mean"]) < 0.002 and row["condition_a_mean"] < 0 and row["condition_b_mean"] < 0 for row in leave_interior.values()),
            {"leave_one_interior_time_slopes": leave_interior},
        ),
        (
            abs(rank_slopes["control_mean"]) < 0.8 and rank_slopes["condition_a_mean"] < -0.9 and rank_slopes["condition_b_mean"] < -0.9,
            {"time_response_spearman": rank_slopes},
        ),
        (
            decline_from_peak["condition_a_mean"] > 0.10 and decline_from_peak["condition_b_mean"] > 0.20 and decline_from_peak["control_mean"] < 0.02,
            {"peak_to_final_decline_fraction": decline_from_peak},
        ),
    )

    dose = pd.read_csv(data / "dose_response.tsv", sep="\t").sort_values("dose").reset_index(drop=True)
    peak_index = int(dose["response_mean"].idxmax())
    leave_one = {}
    for index in range(len(dose)):
        retained = dose.drop(index=index)
        leave_one[str(index)] = {
            "remaining_points": len(retained),
            "interior_peak_identifiable": bool(len(retained) >= 3 and retained["response_mean"].idxmax() not in {retained.index.min(), retained.index.max()}),
        }
    linear = linregress(dose["dose"], dose["response_mean"])
    quadratic = np.polyfit(dose["dose"], dose["response_mean"], 2)
    definitions = {
        "raw": dose["response_mean"].tolist(),
        "relative_to_first": (dose["response_mean"] / dose["response_mean"].iloc[0]).tolist(),
        "relative_to_max": (dose["response_mean"] / dose["response_mean"].max()).tolist(),
    }
    out_b = card(
        {
            "dose_point_count": len(dose),
            "observed_peak_dose": dose.loc[peak_index, "dose"],
            "observed_peak_to_low_dose_ratio": dose.loc[peak_index, "response_mean"] / dose["response_mean"].iloc[0],
            "high_dose_to_observed_peak_ratio": dose["response_mean"].iloc[-1] / dose.loc[peak_index, "response_mean"],
            "leave_one_peak_identifiability_fraction": np.mean([row["interior_peak_identifiable"] for row in leave_one.values()]),
        },
        (
            not any(row["interior_peak_identifiable"] for row in leave_one.values()),
            {"leave_one_dose_fragility": leave_one, "interpretation": "The bounded interior maximum is observed but cannot establish a stable optimum with three points."},
        ),
        (
            linear.slope > 0 and quadratic[0] < 0,
            {"linear_slope": linear.slope, "quadratic_coefficients": quadratic.tolist(), "model_ambiguity": True},
        ),
        (
            all(int(np.argmax(values)) == 1 for values in definitions.values()),
            {"scale_definitions": definitions},
        ),
    )
    return {"A": out_a, "B": out_b}


def connectome(data: Path) -> dict:
    nodes = pd.read_csv(data / "network_node_summary.tsv", sep="\t")
    blocks = {
        str(index): _rho(nodes.iloc[indices]["node_strength"], nodes.iloc[indices]["mean_connection_distance"])
        for index, indices in enumerate(np.array_split(np.arange(len(nodes)), 4))
    }
    leave_decile = {}
    ordered_nodes = nodes.sort_values("region_index").reset_index(drop=True)
    deciles = [ordered_nodes.iloc[indices] for indices in np.array_split(np.arange(len(ordered_nodes)), 10)]
    for index in range(10):
        retained = pd.concat([block for block_index, block in enumerate(deciles) if block_index != index])
        leave_decile[str(index)] = _rho(retained["node_strength"], retained["mean_connection_distance"])
    residual_strength = nodes["node_strength"] - np.polyval(np.polyfit(nodes["region_index"], nodes["node_strength"], 2), nodes["region_index"])
    residual_distance = nodes["mean_connection_distance"] - np.polyval(
        np.polyfit(nodes["region_index"], nodes["mean_connection_distance"], 2), nodes["region_index"]
    )
    out_a = card(
        {
            "node_count": len(nodes),
            "strength_distance_spearman": _rho(nodes["node_strength"], nodes["mean_connection_distance"]),
            "strength_distance_pearson": _pearson(nodes["node_strength"], nodes["mean_connection_distance"]),
            "homophily_distance_spearman": _rho(nodes["mean_homophily"], nodes["mean_connection_distance"]),
        },
        (min(leave_decile.values()) > 0.60, {"leave_one_contiguous_decile_strength_distance_spearman": leave_decile}),
        (min(blocks.values()) > 0.45, {"four_contiguous_region_blocks": blocks}),
        (_rho(residual_strength, residual_distance) > 0.60, {"quadratic_region_index_residual_spearman": _rho(residual_strength, residual_distance)}),
    )

    trim = pd.read_csv(data / "trimming_sensitivity.tsv", sep="\t")
    z = trim[trim["result_kind"] == "assort_z"].copy()
    columns = [name for name in trim.columns if name.startswith("trim_percent_")]
    start = z[columns[0]]
    end = z[columns[-1]]
    leave_row = {}
    for index in z.index:
        retained = z.drop(index=index)
        leave_row[str(index)] = {"start_median": retained[columns[0]].median(), "end_median": retained[columns[-1]].median()}
    early_late = {
        "early_three_median": float(np.median(z[columns[:3]].to_numpy())),
        "late_three_median": float(np.median(z[columns[-3:]].to_numpy())),
    }
    out_b = card(
        {
            "assortativity_series_count": len(z),
            "first_trim_median_z": start.median(),
            "last_trim_median_z": end.median(),
            "decreasing_series_fraction": np.mean(end < start),
            "positive_fraction_first_trim": np.mean(start > 0),
            "positive_fraction_last_trim": np.mean(end > 0),
        },
        (all(row["start_median"] > 0 and row["end_median"] < 0 for row in leave_row.values()), {"leave_one_series_medians": leave_row}),
        (float(start.mean()) > 0 and float(end.mean()) < 0, {"mean_first_and_last_trim_z": {"first": start.mean(), "last": end.mean()}}),
        (early_late["early_three_median"] > 0 and early_late["late_three_median"] < 0, {"three_column_trim_definitions": early_late}),
    )
    return {"A": out_a, "B": out_b}


def immune_context(data: Path) -> dict:
    frame = pd.read_csv(data / "compartment_response_replicates.tsv", sep="\t")

    def ratios(rows: pd.DataFrame, reducer: str = "median") -> pd.Series:
        grouped = rows.groupby(["assay_id", "compartment_id", "endpoint_type", "condition_role"])["response_value"]
        summary = grouped.median() if reducer == "median" else grouped.mean()
        table = summary.unstack("condition_role")
        return table["perturbation"] / table["baseline"]

    base = ratios(frame)
    leave_assay = {
        assay: {"minimum_ratio": ratios(frame[frame["assay_id"] != assay]).min(), "median_ratio": ratios(frame[frame["assay_id"] != assay]).median()}
        for assay in sorted(frame["assay_id"].unique())
    }
    means = ratios(frame, reducer="mean")
    fraction_only = base.xs("endpoint_fraction", level="endpoint_type")
    out_a = card(
        {
            "assay_count": frame["assay_id"].nunique(),
            "compartment_count": frame["compartment_id"].nunique(),
            "minimum_stratum_perturbation_ratio": base.min(),
            "median_stratum_perturbation_ratio": base.median(),
            "maximum_stratum_perturbation_ratio": base.max(),
        },
        (min(row["minimum_ratio"] for row in leave_assay.values()) > 1.20, {"leave_one_complete_assay": leave_assay}),
        (means.min() > 1.20, {"stratum_mean_ratios": means.to_dict()}),
        (fraction_only.min() > 1.20, {"fraction_endpoint_only_ratios": fraction_only.to_dict()}),
    )

    multi = pd.read_csv(data / "multi_condition_response_replicates.tsv", sep="\t")
    table = multi.groupby(["response_class_id", "compound_setting_id"])["response_value"].median().unstack()
    settings = list(table.columns)
    endpoint_folds = {response_class: float(table.loc[response_class, settings[-1]] / table.loc[response_class, settings[0]]) for response_class in table.index}
    class_slopes = {response_class: float(linregress(np.arange(len(settings)), table.loc[response_class].to_numpy()).slope) for response_class in table.index}
    replicate_folds = {}
    for replicate, group in multi.groupby("replicate_id"):
        pivot = group.pivot(index="response_class_id", columns="compound_setting_id", values="response_value")
        replicate_folds[replicate] = {response_class: float(pivot.loc[response_class, settings[-1]] / pivot.loc[response_class, settings[0]]) for response_class in pivot.index}
    rank = {response_class: _rho(np.arange(len(settings)), table.loc[response_class]) for response_class in table.index}
    out_b = card(
        {
            "response_class_A_endpoint_fold": endpoint_folds["response_class_A"],
            "response_class_B_endpoint_fold": endpoint_folds["response_class_B"],
            "response_class_A_linear_setting_slope": class_slopes["response_class_A"],
            "response_class_B_linear_setting_slope": class_slopes["response_class_B"],
        },
        (
            min(row["response_class_A"] for row in replicate_folds.values()) > 3
            and max(row["response_class_B"] for row in replicate_folds.values()) < 1.50,
            {"complete_replicate_endpoint_folds": replicate_folds},
        ),
        (rank["response_class_A"] > 0.9 and abs(rank["response_class_B"]) < 0.5, {"setting_rank_correlations": rank}),
        (endpoint_folds["response_class_A"] > 3 and abs(endpoint_folds["response_class_B"] - 1) < 0.15, {"endpoint_fold_definition": endpoint_folds}),
    )
    return {"A": out_a, "B": out_b}


def nutrient_kinetics(data: Path) -> dict:
    timecourse = pd.read_csv(data / "nutrient_response_timecourse.tsv", sep="\t")

    def time_summary(rows: pd.DataFrame, reducer: str = "median") -> pd.DataFrame:
        grouped = rows.groupby(["time_min", "system_state_id"])["normalized_signaling_response"]
        values = grouped.median() if reducer == "median" else grouped.mean()
        return values.unstack("system_state_id").sort_index()

    base = time_summary(timecourse)
    peak_times = {state: float(base[state].idxmax()) for state in base.columns}
    leave_replicate = {}
    for replicate in sorted(timecourse["replicate_id"].unique()):
        table = time_summary(timecourse[timecourse["replicate_id"] != replicate])
        leave_replicate[replicate] = {state: float(table[state].idxmax()) for state in table.columns}
    mean_table = time_summary(timecourse, reducer="mean")
    auc = {state: float(np.trapezoid(base[state], base.index)) for state in base.columns}
    out_a = card(
        {
            "background_peak_time_min": peak_times["background_state"],
            "perturbed_peak_time_min": peak_times["perturbed_state"],
            "background_peak_to_baseline_fold": base["background_state"].max() / base["background_state"].iloc[0],
            "perturbed_peak_to_baseline_fold": base["perturbed_state"].max() / base["perturbed_state"].iloc[0],
            "perturbed_to_background_ratio_at_5_min": base.loc[5.0, "perturbed_state"] / base.loc[5.0, "background_state"],
            "perturbed_to_background_ratio_at_20_min": base.loc[20.0, "perturbed_state"] / base.loc[20.0, "background_state"],
        },
        (
            all(all(value in {20.0, 30.0} for value in row.values()) for row in leave_replicate.values()),
            {"leave_one_replicate_peak_times": leave_replicate, "interpretation": "The measured maximum is confined to the 20–30 min late window; exact 20 min timing is not stable to replicate removal."},
        ),
        (all(float(mean_table[state].idxmax()) == 20.0 for state in mean_table.columns), {"mean_based_timecourses": mean_table.to_dict()}),
        (min(auc.values()) > 200 and max(auc.values()) / min(auc.values()) < 1.20, {"trapezoidal_timecourse_auc": auc}),
    )

    circulation = pd.read_csv(data / "circulating_input_measurement.tsv", sep="\t")
    uptake = pd.read_csv(data / "tissue_uptake_measurement.tsv", sep="\t")

    def cross_level(reducer: str = "median") -> dict:
        grouped = circulation.groupby(["system_state_id", "input_condition_role"])["measured_abundance_au"]
        c = (grouped.median() if reducer == "median" else grouped.mean()).unstack()
        ugroup = uptake.groupby(["tissue_id", "system_state_id"])["uptake_measurement_au"]
        u = (ugroup.median() if reducer == "median" else ugroup.mean()).unstack()
        return {
            "input_folds": (c["nutrient_input"] / c["pre_input"]).to_dict(),
            "perturbed_to_background_uptake": (u["perturbed_state"] / u["background_state"]).to_dict(),
        }

    base_cross = cross_level()
    leave_replicate_cross = {}
    for replicate in sorted(set(circulation["replicate_id"]) | set(uptake["replicate_id"])):
        c = circulation[circulation["replicate_id"] != replicate]
        u = uptake[uptake["replicate_id"] != replicate]
        cg = c.groupby(["system_state_id", "input_condition_role"])["measured_abundance_au"].median().unstack()
        ug = u.groupby(["tissue_id", "system_state_id"])["uptake_measurement_au"].median().unstack()
        leave_replicate_cross[replicate] = {
            "minimum_input_fold": float((cg["nutrient_input"] / cg["pre_input"]).min()),
            "maximum_perturbed_uptake_ratio": float((ug["perturbed_state"] / ug["background_state"]).max()),
        }
    mean_cross = cross_level(reducer="mean")
    out_b = card(
        {
            "background_input_fold": base_cross["input_folds"]["background_state"],
            "perturbed_input_fold": base_cross["input_folds"]["perturbed_state"],
            "tissue_1_perturbed_to_background_uptake_ratio": base_cross["perturbed_to_background_uptake"]["tissue_1"],
            "tissue_2_perturbed_to_background_uptake_ratio": base_cross["perturbed_to_background_uptake"]["tissue_2"],
        },
        (
            min(row["minimum_input_fold"] for row in leave_replicate_cross.values()) > 5
            and max(row["maximum_perturbed_uptake_ratio"] for row in leave_replicate_cross.values()) < 1.15,
            {"leave_one_replicate_cross_level_metrics": leave_replicate_cross},
        ),
        (
            min(mean_cross["input_folds"].values()) > 5 and max(mean_cross["perturbed_to_background_uptake"].values()) < 1,
            {"mean_based_cross_level_metrics": mean_cross},
        ),
        (
            min(base_cross["input_folds"].values()) > 5 and max(base_cross["perturbed_to_background_uptake"].values()) < 1,
            {"state_normalized_fold_definitions": base_cross},
        ),
    )
    return {"A": out_a, "B": out_b}


def environmental_transfer(data: Path) -> dict:
    units = pd.read_csv(data / "unit_level_observations.tsv", sep="\t")
    summary = pd.read_csv(data / "symptom_assay_surface_summary.tsv", sep="\t")

    def unit_metrics(rows: pd.DataFrame, reducer: str = "median") -> dict:
        grouped = rows.groupby("any_environment_positive")
        assay = grouped["quantitative_assay_value"].median() if reducer == "median" else grouped["quantitative_assay_value"].mean()
        illness = grouped["illness_day"].median() if reducer == "median" else grouped["illness_day"].mean()
        return {
            "assay_negative": float(assay.loc[0]),
            "assay_positive": float(assay.loc[1]),
            "illness_day_negative": float(illness.loc[0]),
            "illness_day_positive": float(illness.loc[1]),
        }

    base = unit_metrics(units)
    leave_unit = {unit: unit_metrics(units[units["unit_id"] != unit]) for unit in units["unit_id"]}
    means = unit_metrics(units, reducer="mean")
    rank = _rho(summary["surface_positive_percent"], summary["quantitative_assay_value"])
    out_a = card(
        {
            "positive_unit_count": int(units["any_environment_positive"].sum()),
            "negative_unit_count": int((units["any_environment_positive"] == 0).sum()),
            "positive_unit_assay_median": base["assay_positive"],
            "negative_unit_assay_median": base["assay_negative"],
            "positive_unit_illness_day_median": base["illness_day_positive"],
            "negative_unit_illness_day_median": base["illness_day_negative"],
            "surface_percent_assay_spearman": rank,
        },
        (
            all(row["assay_positive"] < row["assay_negative"] and row["illness_day_positive"] < row["illness_day_negative"] for row in leave_unit.values()),
            {"leave_one_complete_unit": leave_unit},
        ),
        (
            means["assay_positive"] < means["assay_negative"] and means["illness_day_positive"] < means["illness_day_negative"],
            {"mean_based_unit_metrics": means},
        ),
        (rank < -0.50, {"quantitative_surface_rank_association": rank}),
    )

    sites = pd.read_csv(data / "site_level_observations.tsv", sep="\t")
    site_rates = sites.groupby("site_role_id")["positive"].mean()
    symptom = summary.groupby("symptom_present").agg(
        positive_fraction=("surface_positive_percent", lambda values: float(np.mean(values > 0))),
        assay_median=("quantitative_assay_value", "median"),
    )
    leave_unit_b = {}
    for unit in units["unit_id"]:
        retained_sites = sites[sites["unit_id"] != unit].groupby("site_role_id")["positive"].mean()
        retained_summary = summary[summary["unit_id"] != unit].groupby("symptom_present")["surface_positive_percent"].apply(lambda values: float(np.mean(values > 0)))
        leave_unit_b[unit] = {
            "site_rate_range": float(retained_sites.max() - retained_sites.min()),
            "symptom_positive_fraction_difference": float(abs(retained_summary.loc[1] - retained_summary.loc[0])),
        }
    site_counts = sites.groupby("site_role_id")["positive"].agg(["sum", "count"])
    smoothed_rates = (site_counts["sum"] + 0.5) / (site_counts["count"] + 1)
    out_b = card(
        {
            "site_role_count": sites["site_role_id"].nunique(),
            "site_positive_rate_min": site_rates.min(),
            "site_positive_rate_max": site_rates.max(),
            "site_positive_rate_range": site_rates.max() - site_rates.min(),
            "symptom_positive_fraction_difference": abs(symptom.loc[1, "positive_fraction"] - symptom.loc[0, "positive_fraction"]),
        },
        (
            min(row["site_rate_range"] for row in leave_unit_b.values()) > 0.15
            and max(row["symptom_positive_fraction_difference"] for row in leave_unit_b.values()) < 0.20,
            {"leave_one_unit_site_and_symptom_metrics": leave_unit_b},
        ),
        (smoothed_rates.max() - smoothed_rates.min() > 0.15, {"half_count_smoothed_site_rates": smoothed_rates.to_dict()}),
        (
            abs(symptom.loc[1, "positive_fraction"] - symptom.loc[0, "positive_fraction"]) < 0.10,
            {"unit_level_any_positive_symptom_definition": symptom.to_dict(orient="index")},
        ),
    )
    return {"A": out_a, "B": out_b}


def developmental_interactions(data: Path) -> dict:
    raw = pd.read_csv(data / "stage_indexed_interactions.tsv", sep="\t")
    frame = raw.drop_duplicates().copy()

    def early_late(rows: pd.DataFrame, early_end: int = 5, late_start: int = 12, reducer: str = "median") -> dict:
        early = rows[rows["stage_index"] <= early_end]["interaction_strength_like_value"]
        late = rows[rows["stage_index"] >= late_start]["interaction_strength_like_value"]
        first = early.median() if reducer == "median" else early.mean()
        last = late.median() if reducer == "median" else late.mean()
        return {"early": float(first), "late": float(last), "shift": float(last - first)}

    base = early_late(frame)
    by_pair = {pair: early_late(group) for pair, group in frame.groupby("cell_pair_id")}
    leave_pair = {
        pair: early_late(frame[frame["cell_pair_id"] != pair])
        for pair in sorted(frame["cell_pair_id"].unique())
    }
    definitions = {
        "0_4_vs_13_17": early_late(frame, early_end=4, late_start=13),
        "0_5_vs_12_17": base,
        "0_6_vs_11_17": early_late(frame, early_end=6, late_start=11),
    }
    out_a = card(
        {
            "deduplicated_row_count": len(frame),
            "early_stage_median_strength": base["early"],
            "late_stage_median_strength": base["late"],
            "late_minus_early_median_shift": base["shift"],
            "cell_pair_positive_shift_fraction": np.mean([row["shift"] > 0 for row in by_pair.values()]),
        },
        (min(row["shift"] for row in leave_pair.values()) > 0.75, {"leave_one_cell_pair_shifts": leave_pair}),
        (early_late(frame, reducer="mean")["shift"] > 0.70, {"mean_based_early_late": early_late(frame, reducer="mean")}),
        (min(row["shift"] for row in definitions.values()) > 0.70, {"stage_window_definitions": definitions}),
    )

    def persistence(rows: pd.DataFrame, threshold: float) -> dict:
        selected = rows[rows["association_pvalue"] <= threshold]
        counts = selected.groupby(["interaction_id", "cell_pair_id"])["stage_index"].nunique()
        return {
            "family_count": len(counts),
            "median_stage_count": counts.median(),
            "persistent_fraction_at_least_9_stages": float(np.mean(counts >= 9)),
            "single_stage_fraction": float(np.mean(counts == 1)),
        }

    base_persistence = persistence(frame, 0.001)
    leave_pair_persistence = {
        pair: persistence(frame[frame["cell_pair_id"] != pair], 0.001)
        for pair in sorted(frame["cell_pair_id"].unique())
    }
    thresholds = {str(value): persistence(frame, value) for value in (0.001, 0.005, 0.01)}
    raw_persistence = persistence(raw, 0.001)
    out_b = card(
        {
            "significant_family_count": base_persistence["family_count"],
            "median_significant_stage_count": base_persistence["median_stage_count"],
            "persistent_family_fraction_at_least_9_stages": base_persistence["persistent_fraction_at_least_9_stages"],
            "single_stage_family_fraction": base_persistence["single_stage_fraction"],
        },
        (
            min(row["persistent_fraction_at_least_9_stages"] for row in leave_pair_persistence.values()) > 0.20
            and min(row["single_stage_fraction"] for row in leave_pair_persistence.values()) > 0.25,
            {"leave_one_cell_pair_persistence": leave_pair_persistence},
        ),
        (
            abs(raw_persistence["persistent_fraction_at_least_9_stages"] - base_persistence["persistent_fraction_at_least_9_stages"]) < 0.01,
            {"raw_vs_deduplicated_persistence": {"raw": raw_persistence, "deduplicated": base_persistence}},
        ),
        (
            max(row["persistent_fraction_at_least_9_stages"] for row in thresholds.values())
            - min(row["persistent_fraction_at_least_9_stages"] for row in thresholds.values())
            < 0.02,
            {"association_threshold_definitions": thresholds},
        ),
    )
    return {"A": out_a, "B": out_b}


def nir_emission(data: Path) -> dict:
    aggregate = pd.read_csv(data / "aggregate_state_emission_spectra.tsv", sep="\t")
    dilute = pd.read_csv(data / "dilute_state_emission_spectra.tsv", sep="\t")
    peak_rows = aggregate.loc[
        aggregate.groupby(["design_series", "aggregate_state_id", "composition_fraction"])["relative_emission_au"].idxmax()
    ]
    aggregate_peaks = peak_rows.groupby("design_series")["wavelength_nm"].median()
    dilute_peak_rows = dilute.loc[dilute.groupby("design_series")["relative_emission_au"].idxmax()]
    dilute_peaks = dilute_peak_rows.set_index("design_series")["wavelength_nm"]
    reference_gap = {
        series: float(aggregate_peaks["reference_series"] - aggregate_peaks[series])
        for series in ("series_A", "series_B", "series_C")
    }
    cross_state_gap = {series: float(abs(aggregate_peaks[series] - dilute_peaks[series])) for series in dilute_peaks.index}
    leave_state = {}
    for state in sorted(aggregate["aggregate_state_id"].unique()):
        retained = peak_rows[peak_rows["aggregate_state_id"] != state].groupby("design_series")["wavelength_nm"].median()
        leave_state[state] = {series: float(retained["reference_series"] - retained[series]) for series in ("series_A", "series_B", "series_C")}
    aggregate_weighted = aggregate.assign(weighted=aggregate["wavelength_nm"] * aggregate["relative_emission_au"]).groupby(
        ["design_series", "aggregate_state_id"]
    ).agg(weighted=("weighted", "sum"), intensity=("relative_emission_au", "sum"))
    centroids = (aggregate_weighted["weighted"] / aggregate_weighted["intensity"]).groupby("design_series").median()
    centroid_gap = {series: float(centroids["reference_series"] - centroids[series]) for series in ("series_A", "series_B", "series_C")}
    out_a = card(
        {
            "reference_aggregate_peak_median_nm": aggregate_peaks["reference_series"],
            "minimum_reference_to_design_peak_gap_nm": min(reference_gap.values()),
            "maximum_design_aggregate_to_dilute_peak_gap_nm": max(cross_state_gap.values()),
            "design_series_count": aggregate["design_series"].nunique(),
        },
        (min(value for row in leave_state.values() for value in row.values()) > 80, {"leave_one_aggregate_state_peak_gaps_nm": leave_state}),
        (min(centroid_gap.values()) > 70, {"intensity_weighted_centroid_gaps_nm": centroid_gap}),
        (max(cross_state_gap.values()) < 10, {"aggregate_median_vs_dilute_peak_gaps_nm": cross_state_gap}),
    )

    response = pd.read_csv(data / "long_wavelength_response.tsv", sep="\t")
    by_series = {}
    for series, group in response.groupby("design_series"):
        group = group.sort_values("composition_fraction")
        by_series[series] = {
            "spearman": _rho(group["composition_fraction"], group["long_wavelength_response_au"]),
            "endpoint_fold": float(group["long_wavelength_response_au"].iloc[-1] / group["long_wavelength_response_au"].iloc[0]),
            "theil_sen_slope": float(theilslopes(group["long_wavelength_response_au"], group["composition_fraction"]).slope),
        }
    leave_inner = {}
    for series, group in response.groupby("design_series"):
        ordered = group.sort_values("composition_fraction")
        leave_inner[series] = []
        for index in ordered.index[1:-1]:
            retained = ordered.drop(index=index)
            leave_inner[series].append(_rho(retained["composition_fraction"], retained["long_wavelength_response_au"]))
    out_b = card(
        {
            "series_count": len(by_series),
            "minimum_composition_response_spearman": min(row["spearman"] for row in by_series.values()),
            "minimum_endpoint_fold": min(row["endpoint_fold"] for row in by_series.values()),
            "maximum_endpoint_fold": max(row["endpoint_fold"] for row in by_series.values()),
        },
        (min(value for rows in leave_inner.values() for value in rows) > 0.99, {"leave_one_inner_composition_spearman": leave_inner}),
        (min(row["theil_sen_slope"] for row in by_series.values()) > 0, {"theil_sen_series_slopes": by_series}),
        (min(row["endpoint_fold"] for row in by_series.values()) > 1.10, {"endpoint_fold_definition": by_series}),
    )
    return {"A": out_a, "B": out_b}


def compositional_invariants(data: Path) -> dict:
    counts = pd.read_csv(data / "anonymous_feature_counts.tsv", sep="\t")
    covariates = pd.read_csv(data / "anonymous_sample_covariates.tsv", sep="\t")

    def richness(threshold: float = 1) -> pd.DataFrame:
        selected = counts[counts["relative_count"] >= threshold]
        summary = selected.groupby("sample_id").agg(
            feature_richness=("feature_id", "nunique"),
            retained_total=("relative_count", "sum"),
        )
        return covariates.merge(summary, left_on="sample_id", right_index=True, how="left").fillna(
            {"feature_richness": 0, "retained_total": 0}
        )

    base = richness()
    group_correlations = {
        group: _rho(rows["continuous_covariate"], rows["feature_richness"])
        for group, rows in base.groupby("coarse_group_id")
    }
    parity_correlations = {
        str(parity): _rho(rows["continuous_covariate"], rows["feature_richness"])
        for parity, rows in base.groupby(base["sample_id"].str.extract(r"(\d+)$")[0].astype(int) % 2)
    }
    threshold_correlations = {
        str(threshold): _rho(rows["continuous_covariate"], rows["feature_richness"])
        for threshold in (1, 2, 5, 10)
        for rows in [richness(threshold)]
    }
    out_a = card(
        {
            "covariate_sample_count": len(base),
            "continuous_covariate_richness_spearman": _rho(base["continuous_covariate"], base["feature_richness"]),
            "minimum_within_group_covariate_richness_spearman": min(group_correlations.values()),
            "richness_median_group_A": base[base["coarse_group_id"] == "group_A"]["feature_richness"].median(),
            "richness_median_group_B": base[base["coarse_group_id"] == "group_B"]["feature_richness"].median(),
            "richness_median_group_C": base[base["coarse_group_id"] == "group_C"]["feature_richness"].median(),
        },
        (min(group_correlations.values()) > 0.35, {"complete_coarse_group_correlations": group_correlations}),
        (min(parity_correlations.values()) > 0.40, {"disjoint_sample_parity_correlations": parity_correlations}),
        (min(threshold_correlations.values()) > 0.40, {"minimum_count_richness_definitions": threshold_correlations}),
    )

    scenarios = pd.read_csv(data / "bias_scenario_metrics.tsv", sep="\t")

    def scenario_summary(rows: pd.DataFrame, reducer: str = "median") -> dict:
        grouped = rows.groupby(["scenario_id", "condition_id"])["metric_value"]
        values = grouped.median() if reducer == "median" else grouped.mean()
        table = values.unstack("condition_id")
        return {
            scenario: {
                "condition_2": float(table.loc[scenario, "condition_2"]),
                "condition_3": float(table.loc[scenario, "condition_3"]),
                "gap": float(table.loc[scenario, "condition_3"] - table.loc[scenario, "condition_2"]),
            }
            for scenario in table.index
        }

    base_scenarios = scenario_summary(scenarios)
    metric_numbers = scenarios["metric_id"].str.extract(r"(\d+)$")[0].astype(int)
    halves = {
        "odd_metrics": scenario_summary(scenarios[metric_numbers % 2 == 1]),
        "even_metrics": scenario_summary(scenarios[metric_numbers % 2 == 0]),
    }
    means = scenario_summary(scenarios, reducer="mean")
    nonzero = scenario_summary(scenarios[scenarios["metric_value"] > 0])
    out_b = card(
        {
            "high_variance_condition_gap": base_scenarios["high_variance"]["gap"],
            "small_sample_condition_gap": base_scenarios["small_sample"]["gap"],
            "minimum_condition_3_minus_2_gap": min(row["gap"] for row in base_scenarios.values()),
        },
        (min(row["gap"] for split in halves.values() for row in split.values()) > 0.45, {"disjoint_metric_halves": halves}),
        (min(row["gap"] for row in means.values()) > 0.45, {"mean_based_scenario_summaries": means}),
        (min(row["gap"] for row in nonzero.values()) > 0.40, {"positive_metric_only_definition": nonzero}),
    )
    return {"A": out_a, "B": out_b}


def ode_geometry(data: Path) -> dict:
    trajectories = pd.read_csv(data / "parameter_trajectories.tsv", sep="\t")
    table = trajectories.pivot(index="parameter_index", columns=["solver_family_id", "progress_index"], values="normalised_distance_like_value")

    def family_summary(rows: pd.DataFrame, reducer: str = "median") -> dict:
        result = {}
        for family in sorted(rows["solver_family_id"].unique()):
            pivot = rows[rows["solver_family_id"] == family].pivot(index="parameter_index", columns="progress_index", values="normalised_distance_like_value")
            summary = pivot.median() if reducer == "median" else pivot.mean()
            result[str(family)] = {
                "progress_0": float(summary.loc[0]),
                "progress_2": float(summary.loc[2]),
                "final_to_initial_ratio": float(summary.loc[2] / summary.loc[0]),
                "parameter_decrease_fraction": float(np.mean(pivot[2] < pivot[0])),
            }
        return result

    base = family_summary(trajectories)
    parameter_blocks = {}
    parameter_ids = np.array_split(np.sort(trajectories["parameter_index"].unique()), 5)
    for index, identifiers in enumerate(parameter_ids):
        parameter_blocks[str(index)] = family_summary(trajectories[trajectories["parameter_index"].isin(identifiers)])
    means = family_summary(trajectories, reducer="mean")
    first_step = {}
    for family in (0, 1):
        values = table[family]
        first_step[str(family)] = {
            "progress_1_to_0_ratio": float(values[1].median() / values[0].median()),
            "progress_2_to_0_ratio": float(values[2].median() / values[0].median()),
        }
    out_a = card(
        {
            "solver_0_final_to_initial_median_ratio": base["0"]["final_to_initial_ratio"],
            "solver_1_final_to_initial_median_ratio": base["1"]["final_to_initial_ratio"],
            "solver_0_parameter_decrease_fraction": base["0"]["parameter_decrease_fraction"],
            "solver_1_parameter_decrease_fraction": base["1"]["parameter_decrease_fraction"],
            "final_ratio_solver_1_over_solver_0": base["1"]["final_to_initial_ratio"] / base["0"]["final_to_initial_ratio"],
        },
        (
            max(block["1"]["final_to_initial_ratio"] for block in parameter_blocks.values()) < 0.15
            and min(block["0"]["final_to_initial_ratio"] for block in parameter_blocks.values()) > 0.65,
            {"complete_parameter_blocks": parameter_blocks},
        ),
        (means["1"]["final_to_initial_ratio"] < 0.15 and means["0"]["final_to_initial_ratio"] > 0.65, {"mean_based_family_summary": means}),
        (
            first_step["1"]["progress_2_to_0_ratio"] < first_step["0"]["progress_2_to_0_ratio"] / 5,
            {"progress_endpoint_definitions": first_step},
        ),
    )

    spectrum = pd.read_csv(data / "singular_value_spectrum.tsv", sep="\t")
    pairs = pd.read_csv(data / "measurement_simulation_pairs.tsv", sep="\t")
    errors = np.abs(pairs["simulation_like_value"] - pairs["measurement_like_value"])
    pair_blocks = {}
    for index, indices in enumerate(np.array_split(np.arange(len(pairs)), 5)):
        block = pairs.iloc[indices]
        pair_blocks[str(index)] = {
            "pearson": _pearson(block["measurement_like_value"], block["simulation_like_value"]),
            "median_absolute_error": float(np.median(np.abs(block["simulation_like_value"] - block["measurement_like_value"]))),
        }
    trimmed = pairs[errors <= np.quantile(errors, 0.95)]
    quantile_spans = {
        str(fraction): float(
            spectrum["log_singular_value"].quantile(1 - fraction) - spectrum["log_singular_value"].quantile(fraction)
        )
        for fraction in (0.0, 0.01, 0.05)
    }
    out_b = card(
        {
            "log_singular_value_full_span": spectrum["log_singular_value"].iloc[0] - spectrum["log_singular_value"].iloc[-1],
            "measurement_simulation_pearson": _pearson(pairs["measurement_like_value"], pairs["simulation_like_value"]),
            "measurement_simulation_spearman": _rho(pairs["measurement_like_value"], pairs["simulation_like_value"]),
            "measurement_simulation_median_absolute_error": np.median(errors),
        },
        (min(row["pearson"] for row in pair_blocks.values()) > 0.35, {"five_contiguous_observation_blocks": pair_blocks}),
        (_pearson(trimmed["measurement_like_value"], trimmed["simulation_like_value"]) > 0.78, {"five_percent_error_trimmed_pearson": _pearson(trimmed["measurement_like_value"], trimmed["simulation_like_value"])}),
        (min(quantile_spans.values()) > 15, {"singular_spectrum_quantile_spans": quantile_spans}),
    )
    return {"A": out_a, "B": out_b}


def surveillance_geometry(data: Path) -> dict:
    panels = {index: pd.read_csv(data / f"observation_set_{index}.tsv", sep="\t") for index in range(1, 5)}
    panel1 = panels[1]
    outcomes1 = list(panel1.columns[2:])
    rhos1 = {name: _rho(panel1["numeric_feature_1"], panel1[name]) for name in outcomes1}
    leave_rows = {}
    for index in range(len(panel1)):
        retained = panel1.drop(index=panel1.index[index])
        leave_rows[str(index)] = min(-_rho(retained["numeric_feature_1"], retained[name]) for name in outcomes1)
    slopes = {name: float(linregress(panel1["numeric_feature_1"], panel1[name]).slope) for name in outcomes1}
    endpoint_ratios = {name: float(panel1[name].iloc[-1] / panel1[name].iloc[0]) for name in outcomes1}
    out_a = card(
        {
            "panel_1_response_count": len(outcomes1),
            "panel_1_maximum_response_spearman": max(rhos1.values()),
            "panel_1_minimum_response_spearman": min(rhos1.values()),
            "panel_1_mean_endpoint_ratio": np.mean(list(endpoint_ratios.values())),
        },
        (min(leave_rows.values()) > 0.99, {"leave_one_ordered_row_minimum_negative_rank_strength": leave_rows}),
        (max(slopes.values()) < 0, {"panel_1_linear_slopes": slopes}),
        (max(endpoint_ratios.values()) < 0.90, {"panel_1_endpoint_ratios": endpoint_ratios}),
    )

    cross = {}
    for index in (2, 3, 4):
        frame = panels[index]
        coordinate = "numeric_feature_1"
        outcomes = [name for name in frame.columns if name not in {"observation_index", coordinate}]
        cross[str(index)] = {
            name: {
                "spearman": _rho(frame[coordinate], frame[name]),
                "endpoint_ratio": float(frame[name].iloc[-1] / frame[name].iloc[0]),
                "slope": float(linregress(frame[coordinate], frame[name]).slope),
            }
            for name in outcomes
        }
    flat = [row for panel in cross.values() for row in panel.values()]
    negative_count = sum(row["spearman"] < 0 for row in flat)
    half_checks = {}
    for index in (2, 3, 4):
        frame = panels[index]
        coordinate = "numeric_feature_1"
        outcomes = [name for name in frame.columns if name not in {"observation_index", coordinate}]
        half_checks[str(index)] = [
            sum(_rho(part[coordinate], part[name]) < 0 for name in outcomes)
            for part in (frame.iloc[: len(frame) // 2], frame.iloc[len(frame) // 2 :])
        ]
    slope_negative_count = sum(row["slope"] < 0 for row in flat)
    endpoint_negative_count = sum(row["endpoint_ratio"] < 1 for row in flat)
    out_b = card(
        {
            "cross_panel_response_count": len(flat),
            "negative_rank_response_count": negative_count,
            "positive_rank_exception_count": len(flat) - negative_count,
            "minimum_cross_panel_spearman": min(row["spearman"] for row in flat),
            "maximum_cross_panel_spearman": max(row["spearman"] for row in flat),
        },
        (
            all(sum(rows) >= 1 for rows in half_checks.values()) and sum(sum(rows) for rows in half_checks.values()) >= 16,
            {"contiguous_half_negative_counts": half_checks},
        ),
        (slope_negative_count >= 10, {"negative_linear_slope_count": slope_negative_count, "cross_panel_metrics": cross}),
        (endpoint_negative_count >= 10, {"endpoint_decline_count": endpoint_negative_count, "cross_panel_metrics": cross}),
    )
    return {"A": out_a, "B": out_b}


def anomalous_trajectories(data: Path) -> dict:
    archive = np.load(data / "anonymous_trajectories.npz")
    trajectories = archive["trajectories"]

    def scaling(max_lag: int, rows: np.ndarray = trajectories) -> np.ndarray:
        lags = np.arange(1, max_lag + 1)
        values = []
        for trajectory in rows:
            msd = np.asarray([np.mean((trajectory[lag:] - trajectory[:-lag]) ** 2) for lag in lags])
            values.append(linregress(np.log(lags), np.log(msd + 1e-30)).slope)
        return np.asarray(values)

    base = scaling(32)
    halves = {
        "first_120": scaling(32, trajectories[:120]),
        "last_120": scaling(32, trajectories[120:]),
    }
    definitions = {str(lag): scaling(lag) for lag in (16, 24, 32, 48)}
    robust = []
    lags = np.arange(1, 33)
    for trajectory in trajectories:
        msd = np.asarray([np.mean((trajectory[lag:] - trajectory[:-lag]) ** 2) for lag in lags])
        robust.append(theilslopes(np.log(msd + 1e-30), np.log(lags)).slope)
    out_a = card(
        {
            "trajectory_count": len(trajectories),
            "median_time_averaged_scaling_exponent": np.median(base),
            "fraction_scaling_exponent_below_0_75": np.mean(base < 0.75),
            "fraction_scaling_exponent_above_1_25": np.mean(base > 1.25),
            "scaling_exponent_interquartile_range": np.percentile(base, 75) - np.percentile(base, 25),
        },
        (
            min(np.mean(values < 0.75) for values in halves.values()) > 0.35 and max(np.mean(values > 1.25) for values in halves.values()) < 0.08,
            {name: {"median": np.median(values), "fraction_below_0_75": np.mean(values < 0.75), "fraction_above_1_25": np.mean(values > 1.25)} for name, values in halves.items()},
        ),
        (np.mean(np.asarray(robust) < 0.75) > 0.35, {"theil_sen_scaling_summary": {"median": np.median(robust), "fraction_below_0_75": np.mean(np.asarray(robust) < 0.75)}}),
        (
            min(np.mean(values < 0.75) for values in definitions.values()) > 0.30
            and max(np.mean(values > 1.25) for values in definitions.values()) < 0.08,
            {lag: {"median": np.median(values), "fraction_below_0_75": np.mean(values < 0.75), "fraction_above_1_25": np.mean(values > 1.25)} for lag, values in definitions.items()},
        ),
    )

    increments = np.diff(trajectories, axis=1)

    def local_contrast(window: int, rows: np.ndarray = increments) -> dict:
        scores = []
        locations = []
        for values in rows:
            contrasts = []
            for split in range(window, len(values) - window):
                left = np.var(values[split - window : split])
                right = np.var(values[split : split + window])
                contrasts.append(abs(np.log((right + 1e-12) / (left + 1e-12))))
            best = int(np.argmax(contrasts))
            scores.append(contrasts[best])
            locations.append(best + window)
        scores = np.asarray(scores)
        locations = np.asarray(locations)
        return {
            "median_max_log_variance_contrast": np.median(scores),
            "fraction_above_log_2": np.mean(scores > np.log(2)),
            "median_location": np.median(locations),
            "location_iqr": np.percentile(locations, 75) - np.percentile(locations, 25),
        }

    base_contrast = local_contrast(16)
    sample_contrast = {"first_120": local_contrast(16, increments[:120]), "last_120": local_contrast(16, increments[120:])}
    window_definitions = {str(window): local_contrast(window) for window in (8, 12, 16, 20)}
    signed_definitions = {}
    for window in (12, 16, 20):
        result = local_contrast(window)
        signed_definitions[str(window)] = result
    out_b = card(
        {
            "median_max_local_log_variance_contrast": base_contrast["median_max_log_variance_contrast"],
            "fraction_trajectories_with_local_variance_contrast_above_twofold": base_contrast["fraction_above_log_2"],
            "median_candidate_change_location": base_contrast["median_location"],
            "candidate_change_location_iqr": base_contrast["location_iqr"],
        },
        (min(row["fraction_above_log_2"] for row in sample_contrast.values()) > 0.90, {"disjoint_trajectory_halves": sample_contrast}),
        (min(row["fraction_above_log_2"] for row in window_definitions.values()) > 0.90, {"local_variance_window_methods": window_definitions}),
        (
            min(row["median_max_log_variance_contrast"] for row in signed_definitions.values()) > 1.2,
            {"candidate_change_definition_windows": signed_definitions, "interpretation": "A local contrast is a candidate regime boundary, not proof of an abrupt generative change point."},
        ),
    )
    return {"A": out_a, "B": out_b}


def calcium_regions(data: Path) -> dict:
    frame = pd.read_csv(data / "cellular_calcium_response_summaries.tsv", sep="\t")
    shared_regions = ["region_1", "region_2"]

    def width_ratios(rows: pd.DataFrame, reducer: str = "median") -> dict:
        grouped = rows[rows["anatomical_region_id"].isin(shared_regions)].groupby(
            ["anatomical_region_id", "condition"]
        )["mean_peak_width_s"]
        values = grouped.median() if reducer == "median" else grouped.mean()
        table = values.unstack("condition")
        return (table["receptor_stimulated"] / table["baseline"]).to_dict()

    base = width_ratios(frame)
    batch_holdouts = {}
    for batch in sorted(frame["recording_batch_id"].unique()):
        ratios = width_ratios(frame[frame["recording_batch_id"] != batch])
        batch_holdouts[batch] = ratios
    parity = {str(offset): width_ratios(frame.iloc[offset::2]) for offset in (0, 1)}
    means = width_ratios(frame, reducer="mean")
    out_a = card(
        {
            "region_1_receptor_to_baseline_peak_width_ratio": base["region_1"],
            "region_2_receptor_to_baseline_peak_width_ratio": base["region_2"],
            "minimum_shared_region_peak_width_ratio": min(base.values()),
            "cell_observation_count": len(frame),
        },
        (min(value for row in batch_holdouts.values() for value in row.values()) > 2.5, {"leave_one_recording_batch_width_ratios": batch_holdouts}),
        (min(means.values()) > 2.5, {"mean_based_width_ratios": means}),
        (min(value for row in parity.values() for value in row.values()) > 2.5, {"disjoint_observation_parity_width_ratios": parity}),
    )

    stimulated = frame[frame["condition"] == "receptor_stimulated"]

    def region_metrics(rows: pd.DataFrame, reducer: str = "median") -> dict:
        grouped = rows.groupby("anatomical_region_id")
        width = grouped["mean_peak_width_s"].median() if reducer == "median" else grouped["mean_peak_width_s"].mean()
        frequency = grouped["mean_peak_frequency_hz"].median() if reducer == "median" else grouped["mean_peak_frequency_hz"].mean()
        prominence = grouped["mean_peak_prominence_dff0"].median() if reducer == "median" else grouped["mean_peak_prominence_dff0"].mean()
        return {
            "region_2_to_1_width_ratio": float(width["region_2"] / width["region_1"]),
            "region_1_to_2_frequency_ratio": float(frequency["region_1"] / frequency["region_2"]),
            "region_2_to_1_prominence_ratio": float(prominence["region_2"] / prominence["region_1"]),
        }

    region_base = region_metrics(stimulated)
    stimulated_batch_holdouts = {
        batch: region_metrics(stimulated[stimulated["recording_batch_id"] != batch])
        for batch in sorted(stimulated["recording_batch_id"].unique())
        if len(stimulated[stimulated["recording_batch_id"] != batch]) > 0
    }
    region_means = region_metrics(stimulated, reducer="mean")
    region_parity = {str(offset): region_metrics(stimulated.iloc[offset::2]) for offset in (0, 1)}
    out_b = card(
        {
            **region_base,
            "stimulated_region_count": stimulated["anatomical_region_id"].nunique(),
        },
        (
            min(row["region_2_to_1_width_ratio"] for row in stimulated_batch_holdouts.values()) > 1.5
            and min(row["region_1_to_2_frequency_ratio"] for row in stimulated_batch_holdouts.values()) > 1.5,
            {"leave_one_recording_batch_region_metrics": stimulated_batch_holdouts},
        ),
        (
            region_means["region_2_to_1_width_ratio"] > 1.40 and region_means["region_1_to_2_frequency_ratio"] > 1.5,
            {"mean_based_region_metrics": region_means},
        ),
        (
            min(row["region_2_to_1_width_ratio"] for row in region_parity.values()) > 1.5
            and min(row["region_1_to_2_frequency_ratio"] for row in region_parity.values()) > 1.5,
            {"disjoint_observation_parity_region_metrics": region_parity},
        ),
    )
    return {"A": out_a, "B": out_b}


def spatial_stability(data: Path) -> dict:
    tracked = pd.read_csv(data / "tracked_unit_spatial_stability.tsv", sep="\t")
    metrics = ["new_environment_stability_2", "new_environment_stability_3", "novel_vs_repeated_change"]

    def contrasts(rows: pd.DataFrame, reducer: str = "median") -> dict:
        grouped = rows.groupby("condition")[metrics]
        values = grouped.median() if reducer == "median" else grouped.mean()
        return {
            metric: {
                "matrix_perturbed": float(values.loc["matrix_perturbed", metric]),
                "reference": float(values.loc["reference", metric]),
                "difference": float(values.loc["matrix_perturbed", metric] - values.loc["reference", metric]),
            }
            for metric in metrics
        }

    base = contrasts(tracked)
    leave_animal = {
        animal: contrasts(tracked[tracked["animal_id"] != animal])
        for animal in sorted(tracked["animal_id"].unique())
        if tracked[tracked["animal_id"] != animal]["condition"].nunique() == 2
    }
    animal_medians = tracked.groupby(["animal_id", "condition"])[metrics].median().reset_index()
    animal_level = contrasts(animal_medians)
    composite_definitions = {}
    for reducer in ("mean", "median"):
        composite = tracked[["new_environment_stability_2", "new_environment_stability_3"]]
        values = composite.mean(axis=1) if reducer == "mean" else composite.median(axis=1)
        summarized = tracked.assign(composite_new_environment_stability=values).groupby("condition")[
            "composite_new_environment_stability"
        ].median()
        composite_definitions[reducer] = {
            "matrix_perturbed": float(summarized["matrix_perturbed"]),
            "reference": float(summarized["reference"]),
            "difference": float(summarized["matrix_perturbed"] - summarized["reference"]),
        }
    out_a = card(
        {
            "new_environment_stability_2_condition_difference": base["new_environment_stability_2"]["difference"],
            "new_environment_stability_3_condition_difference": base["new_environment_stability_3"]["difference"],
            "novel_vs_repeated_change_condition_difference": base["novel_vs_repeated_change"]["difference"],
            "tracked_unit_count": len(tracked),
        },
        (
            max(row["new_environment_stability_2"]["difference"] for row in leave_animal.values()) < -0.10
            and max(row["new_environment_stability_3"]["difference"] for row in leave_animal.values()) < -0.18,
            {"leave_one_animal_condition_contrasts": leave_animal},
        ),
        (
            animal_level["new_environment_stability_2"]["difference"] < -0.10
            and animal_level["new_environment_stability_3"]["difference"] < -0.18,
            {"animal_equal_condition_contrasts": animal_level},
        ),
        (
            max(row["difference"] for row in composite_definitions.values()) < -0.15,
            {"two_new_environment_composite_definitions": composite_definitions},
        ),
    )

    sessions = pd.read_csv(data / "unit_session_metrics.tsv", sep="\t")
    animal_rate = sessions.groupby(["animal_id", "condition"])["average_rate_hz"].median().reset_index()
    rate_medians = animal_rate.groupby("condition")["average_rate_hz"].median()
    rate_ratio = float(rate_medians["matrix_perturbed"] / rate_medians["reference"])
    leave_animal_rate = {}
    for animal in sorted(animal_rate["animal_id"].unique()):
        retained = animal_rate[animal_rate["animal_id"] != animal].groupby("condition")["average_rate_hz"].median()
        if len(retained) == 2:
            leave_animal_rate[animal] = float(retained["matrix_perturbed"] / retained["reference"])
    trimmed_rate = sessions.groupby(["animal_id", "condition"])["average_rate_hz"].mean().reset_index().groupby("condition")["average_rate_hz"].median()
    mean_rate_ratio = float(trimmed_rate["matrix_perturbed"] / trimmed_rate["reference"])
    stability_difference = base["new_environment_stability_3"]["difference"]
    out_b = card(
        {
            "animal_equal_average_rate_condition_ratio": rate_ratio,
            "new_environment_stability_3_condition_difference": stability_difference,
            "absolute_rate_ratio_deviation_from_one": abs(rate_ratio - 1),
        },
        (
            max(abs(value - 1) for value in leave_animal_rate.values()) < 0.25,
            {"leave_one_animal_rate_ratios": leave_animal_rate},
        ),
        (abs(mean_rate_ratio - 1) < 0.20 and stability_difference < -0.18, {"animal_mean_rate_ratio": mean_rate_ratio, "stability_difference": stability_difference}),
        (
            abs(rate_ratio - 1) < 0.15 and base["novel_vs_repeated_change"]["difference"] < -0.10,
            {"rate_stability_dissociation_definition": {"rate_ratio": rate_ratio, "novel_change_difference": base["novel_vs_repeated_change"]["difference"]}},
        ),
    )
    return {"A": out_a, "B": out_b}


def response_sequence(data: Path) -> dict:
    latency = pd.read_csv(data / "condition_channel_latency.tsv", sep="\t")
    table = latency.pivot(index="channel_index", columns="condition_id", values="latency_like_value")
    medians = table.median()
    ordered_fraction = float(np.mean((table["condition_1"] < table["condition_2"]) & (table["condition_2"] < table["condition_3"])))
    channel_halves = {
        str(offset): {
            "medians": table.iloc[offset::2].median().to_dict(),
            "ordered_fraction": float(
                np.mean(
                    (table.iloc[offset::2]["condition_1"] < table.iloc[offset::2]["condition_2"])
                    & (table.iloc[offset::2]["condition_2"] < table.iloc[offset::2]["condition_3"])
                )
            ),
        }
        for offset in (0, 1)
    }
    means = table.mean()
    trimmed = table.apply(lambda values: values[(values >= values.quantile(0.05)) & (values <= values.quantile(0.95))].median())
    out_a = card(
        {
            "condition_1_median_latency": medians["condition_1"],
            "condition_2_median_latency": medians["condition_2"],
            "condition_3_median_latency": medians["condition_3"],
            "strict_channelwise_order_fraction": ordered_fraction,
            "condition_3_minus_1_median_latency": medians["condition_3"] - medians["condition_1"],
        },
        (
            min(row["ordered_fraction"] for row in channel_halves.values()) > 0.90,
            {"disjoint_channel_halves": channel_halves},
        ),
        (means["condition_1"] < means["condition_2"] < means["condition_3"], {"condition_mean_latencies": means.to_dict()}),
        (trimmed["condition_1"] < trimmed["condition_2"] < trimmed["condition_3"], {"five_percent_trimmed_condition_medians": trimmed.to_dict()}),
    )

    spans = pd.read_csv(data / "condition_sequence_span.tsv", sep="\t")
    span_table = spans.groupby("condition_id")["sequence_span_like_value"]
    span_medians = span_table.median()
    replicate_halves = {
        str(offset): spans.iloc[offset::2].groupby("condition_id")["sequence_span_like_value"].median().to_dict()
        for offset in (0, 1)
    }
    span_means = span_table.mean()
    persistence = pd.read_csv(data / "condition_channel_persistence.tsv", sep="\t")
    persistence_order = {
        str(panel): group.groupby("condition_id")["decay_speed_like_value"].median().sort_values().to_dict()
        for panel, group in persistence.groupby("comparison_panel_index")
    }
    out_b = card(
        {
            "condition_1_median_sequence_span": span_medians.loc[1],
            "condition_2_median_sequence_span": span_medians.loc[2],
            "condition_3_median_sequence_span": span_medians.loc[3],
            "condition_3_to_1_sequence_span_ratio": span_medians.loc[3] / span_medians.loc[1],
            "persistence_panel_count": persistence["comparison_panel_index"].nunique(),
        },
        (
            all(row[1] < row[2] < row[3] for row in replicate_halves.values()),
            {"disjoint_sequence_replicate_halves": replicate_halves},
        ),
        (span_means.loc[1] < span_means.loc[2] < span_means.loc[3], {"condition_mean_sequence_spans": span_means.to_dict()}),
        (
            all(max(values.values()) - min(values.values()) > 15 for values in persistence_order.values()),
            {"panel_specific_persistence_medians": persistence_order},
        ),
    )
    return {"A": out_a, "B": out_b}


def temporal_geometry(data: Path) -> dict:
    recordings = {
        "A": pd.read_csv(data / "event_recording_set_a.tsv", sep="\t"),
        "B": pd.read_csv(data / "event_recording_set_b.tsv", sep="\t"),
    }
    base = {
        name: {
            "spearman": _rho(frame["event_relative_time"], frame["pre_event_signal"]),
            "pearson": _pearson(frame["event_relative_time"], frame["pre_event_signal"]),
        }
        for name, frame in recordings.items()
    }
    halves = {}
    for name, frame in recordings.items():
        halves[name] = [
            _rho(part["event_relative_time"], part["pre_event_signal"])
            for part in (frame.iloc[: len(frame) // 2], frame.iloc[len(frame) // 2 :])
        ]
    trimmed = {}
    for name, frame in recordings.items():
        low, high = frame["event_relative_time"].quantile([0.05, 0.95])
        retained = frame[frame["event_relative_time"].between(low, high)]
        trimmed[name] = _rho(retained["event_relative_time"], retained["pre_event_signal"])
    sign_groups = {
        name: frame.assign(nonnegative=frame["event_relative_time"] >= 0).groupby("nonnegative")["pre_event_signal"].median().to_dict()
        for name, frame in recordings.items()
    }
    out_a = card(
        {
            "recording_A_time_signal_spearman": base["A"]["spearman"],
            "recording_B_time_signal_spearman": base["B"]["spearman"],
            "recording_A_time_signal_pearson": base["A"]["pearson"],
            "recording_B_time_signal_pearson": base["B"]["pearson"],
        },
        (max(value for rows in halves.values() for value in rows) < -0.20, {"contiguous_event_halves": halves}),
        (max(trimmed.values()) < -0.20, {"five_percent_time_trimmed_spearman": trimmed}),
        (
            all(values[False] > values[True] for values in sign_groups.values()),
            {"negative_vs_nonnegative_event_time_signal_medians": sign_groups},
        ),
    )

    mean_latency = pd.read_csv(data / "unit_distance_mean_latency.tsv", sep="\t")
    variability = pd.read_csv(data / "unit_distance_latency_variability.tsv", sep="\t")
    distance = {
        "mean_latency_spearman": _rho(mean_latency["pair_distance"], mean_latency["mean_event_latency"]),
        "variability_spearman": _rho(variability["pair_distance"], variability["event_latency_sd"]),
        "mean_latency_pearson": _pearson(mean_latency["pair_distance"], mean_latency["mean_event_latency"]),
        "variability_pearson": _pearson(variability["pair_distance"], variability["event_latency_sd"]),
    }
    parity = {
        str(offset): {
            "mean_latency_spearman": _rho(mean_latency.iloc[offset::2]["pair_distance"], mean_latency.iloc[offset::2]["mean_event_latency"]),
            "variability_spearman": _rho(variability.iloc[offset::2]["pair_distance"], variability.iloc[offset::2]["event_latency_sd"]),
        }
        for offset in (0, 1)
    }
    trims = {}
    for count in (1, 2, 3, 4):
        left = mean_latency.sort_values("pair_distance").iloc[count:-count]
        right = variability.sort_values("pair_distance").iloc[count:-count]
        trims[str(count)] = {
            "mean_latency_spearman": _rho(left["pair_distance"], left["mean_event_latency"]),
            "variability_spearman": _rho(right["pair_distance"], right["event_latency_sd"]),
        }
    out_b = card(
        {
            **distance,
            "unit_pair_count": len(mean_latency),
        },
        (
            max(abs(value) for row in parity.values() for value in row.values()) < 0.30,
            {"disjoint_unit_parity_correlations": parity},
        ),
        (max(abs(distance["mean_latency_pearson"]), abs(distance["variability_pearson"])) < 0.25, {"linear_distance_correlations": distance}),
        (
            max(abs(value) for row in trims.values() for value in row.values()) < 0.30,
            {"distance_endpoint_trim_definitions": trims, "interpretation": "The supplied unit summaries bound a simple monotone distance explanation; they do not prove absence of nonlinear spatial organization."},
        ),
    )
    return {"A": out_a, "B": out_b}


MULTIMODAL_HANDLERS = {
    "Astronomy_05_stellar_spectral_boundary": stellar_spectra,
    "Chemistry_08_catalysis_response_landscape": catalysis_landscape,
    "EarthScience_05_ocean_thermal_coupling": ocean_record,
    "Energy_03_photoreduction_formulations": photoreduction,
    "Energy_06_interface_response_boundaries": interface_response,
    "Energy_07_biohybrid_operating_tradeoffs": biohybrid,
    "Information_04_connectome_assortative_structure": connectome,
    "Life_06_contextual_immune_response": immune_context,
    "Life_07_nutrient_response_kinetics": nutrient_kinetics,
    "Life_08_environmental_transfer_boundary": environmental_transfer,
    "Life_09_developmental_interaction_structure": developmental_interactions,
    "Material_06_nir_emission_structure": nir_emission,
    "Math_02_compositional_invariants": compositional_invariants,
    "Math_03_ode_parameter_geometry": ode_geometry,
    "Math_04_surveillance_spectral_geometry": surveillance_geometry,
    "Math_05_anomalous_diffusion_change_points": anomalous_trajectories,
    "Neuro_02_calcium_response_regions": calcium_regions,
    "Neuro_03_spatial_code_stability": spatial_stability,
    "Neuro_04_response_sequence_boundary": response_sequence,
    "Neuro_05_temporal_geometry": temporal_geometry,
}
