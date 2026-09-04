#!/usr/bin/env python3
"""Physics_01 的冻结 EPR/NMR 复算与扰动核心。

EPR 路径直接移植作者 ``uwb_eval.m`` 的解析信号、下变频、Chebyshev
回波窗与固定参考相位步骤；NMR 路径直接移植八步相位循环，并用赛前冻结
的独立参考波形做复数匹配投影。Judge 只执行，不临时选择处理方法。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve, hilbert, windows


SOURCE_SHA256 = {
    "20240424_1547_dnpdata_nmr_only.mat": "21d8ddead3568549e78e863da1991e2aaec29a5869b6e5f7e71c972c8521bfd0",
    "20240522_125527_dnp_echo_pp_length_2.mat": "11ba1cc18b975b1d46efa79a3b1bf960fabb7b9a890fdf74b591aa238a5d3935",
    "20240522_130805_dnp_echo_pp_length_2.mat": "2586271e3721e1ac1d547c3f9e3cae36b319253d1dc1c6104bf391b50766341b",
    "20240522_130914_dnp_echo_pp_length_2.mat": "8061a82531bb740b92eefeac81147b853213d9bbc5df5d40499239ca8560a7bd",
    "20240522_131018_dnp_echo_pp_length_2.mat": "a7f2cd0f75ef88669a4c524d52762a0e4e1e3566c3d11de8c28c415de3c6c3d0",
    "20240522_131142_dnp_echo_pp_length_2.mat": "7c3875bc6a46a935c07c03375a21caedaa3284eaa65f05ac11c3e683f2856ef3",
    "20240522_131404_dnp_echo_pp_length_2.mat": "ef10c2bf4789109be0a64665e62520cb76983ef3563ca6f83061d870b658b900",
    "20240522_1404_dnpdata_dnp_contact_swp.mat": "2ace5b6c0593bda645fd2e018608aa9718f6364bcb39d2d0eb56bee8387062cd",
    "20240522_1434_dnpdata_dnp_contact_swp.mat": "a88a16a44385647ad2aafa228be866b9b29a8258d8a98b948995d13436e26294",
    "20240522_1505_dnpdata_dnp_contact_swp.mat": "a287ebd999dabfaa700e9bdd4f8069df5421ab609ce4f9a6962751a9fa59c852",
    "20240522_1535_dnpdata_dnp_contact_swp.mat": "77240a4a63263f31a25fa9ae9a2c8e067631659b0851b8a4cf7f15e5e80e23f5",
    "20240523_0954_dnpdata_dnp_contact_swp.mat": "e151b6fd905f17fe498f9d10a2a1a977f931865ee50681b8df412a02dce00872",
    "20240523_1308_dnpdata_dnp_contact_swp.mat": "501c5b027290bc1051d8d61a2b3dab34354f3e786c0e1f15c3e1ba038d79ca2a",
}
EPR_FILES = (
    ("20240522_125527_dnp_echo_pp_length_2.mat", 0.0),
    ("20240522_130805_dnp_echo_pp_length_2.mat", 132.0),
    ("20240522_130914_dnp_echo_pp_length_2.mat", 220.0),
    ("20240522_131018_dnp_echo_pp_length_2.mat", 440.0),
    ("20240522_131142_dnp_echo_pp_length_2.mat", 880.0),
    ("20240522_131404_dnp_echo_pp_length_2.mat", 2200.0),
)
NMR_FILES = (
    ("20240523_1308_dnpdata_dnp_contact_swp.mat", 0.0),
    ("20240522_1404_dnpdata_dnp_contact_swp.mat", 132.0),
    ("20240522_1434_dnpdata_dnp_contact_swp.mat", 220.0),
    ("20240522_1505_dnpdata_dnp_contact_swp.mat", 440.0),
    ("20240522_1535_dnpdata_dnp_contact_swp.mat", 880.0),
    ("20240523_0954_dnpdata_dnp_contact_swp.mat", 2200.0),
)
REFERENCE_FILE = "20240424_1547_dnpdata_nmr_only.mat"


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
    for filename, expected in SOURCE_SHA256.items():
        actual = _sha256(root / "raw" / filename)
        if actual != expected:
            raise RuntimeError(f"source data drift: {filename}: {actual} != {expected}")


def _mat(root: Path, filename: str) -> dict:
    return loadmat(root / "raw" / filename, squeeze_me=True, struct_as_record=False)


def epr_curve(root: Path, filename: str, window_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    saved = _mat(root, filename)
    experiment = saved["dnp_echo"]
    raw = np.asarray(saved["dta_001"], dtype=float)
    axis = np.asarray(np.atleast_1d(experiment.parvars)[1].axis, dtype=float)
    if raw.shape[1] != len(axis):
        raise RuntimeError("EPR indirect axis does not match raw traces")

    sampling_rate = float(saved["conf"].std.dig_rate)
    detection = np.atleast_1d(experiment.events)[int(experiment.det_event) - 1]
    detection_frequency = float(detection.det_frq)
    time = np.arange(raw.shape[0], dtype=float) / sampling_rate
    downconverted = np.conj(hilbert(raw, axis=0)) * np.exp(
        -2j * np.pi * time[:, None] * detection_frequency
    )

    full_window = windows.chebwin(raw.shape[0], at=100)
    reference_index = int(np.argmax(np.abs(np.sum(downconverted * full_window[:, None], axis=0))))
    matched_shape = windows.chebwin(min(100, raw.shape[0]), at=100)
    center = int(
        np.argmax(convolve(np.abs(downconverted[:, reference_index]), matched_shape, mode="same"))
    )
    start = center - window_length // 2 + 1
    stop = start + window_length
    if start < 0 or stop > raw.shape[0]:
        raise RuntimeError("frozen EPR evaluation window is out of range")
    window = windows.chebwin(window_length, at=100)
    selected = downconverted[start:stop]
    phase = np.angle(np.sum(selected * window[:, None], axis=0)[reference_index])
    integrated = np.sum(
        selected * window[:, None] * np.exp(-1j * phase), axis=0
    ) / np.sum(window)
    return axis, np.real(integrated)


def _near_index(axis: np.ndarray, expected: float) -> int:
    return int(np.argmin(np.abs(axis - expected)))


def _local_indices(axis: np.ndarray, expected: float, radius: float = 88.0) -> np.ndarray:
    indices = np.flatnonzero(np.abs(axis - expected) <= radius)
    if not len(indices):
        raise RuntimeError("expected echo is outside measured axis")
    return indices


def epr_summary(root: Path, window_length: int = 256) -> dict:
    curves = []
    for filename, inversion in EPR_FILES:
        axis, values = epr_curve(root, filename, window_length=window_length)
        curves.append((inversion, axis + inversion, values))
    reference_peak = float(np.max(np.abs(curves[0][2])))
    conditions = []
    for inversion, absolute_axis, values in curves[1:]:
        expected = 2.0 * inversion
        candidates = _local_indices(absolute_axis, expected)
        detected_index = int(candidates[np.argmax(values[candidates])])
        exact_index = _near_index(absolute_axis, expected)
        conditions.append(
            {
                "inversion_ns": inversion,
                "expected_echo_ns": expected,
                "detected_echo_ns": float(absolute_axis[detected_index]),
                "timing_error_ns": float(abs(absolute_axis[detected_index] - expected)),
                "local_peak_fraction": float(values[detected_index] / reference_peak),
                "expected_sample_fraction": float(values[exact_index] / reference_peak),
            }
        )
    local_fractions = np.asarray([row["local_peak_fraction"] for row in conditions])
    return {
        "reference_peak": reference_peak,
        "window_length": window_length,
        "conditions": conditions,
        "max_abs_timing_error_ns": max(row["timing_error_ns"] for row in conditions),
        "median_abs_timing_error_ns": float(np.median([row["timing_error_ns"] for row in conditions])),
        "short_echo_fraction": float(local_fractions[0]),
        "long_echo_fraction": float(local_fractions[-1]),
        "echo_condition_count": len(conditions),
    }


def _phase_cycled_nmr(root: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    saved = _mat(root, filename)
    experiment = saved["nmr_exp"]
    phases = np.asarray(experiment.det_phases, dtype=float) / 180.0 * np.pi
    names = sorted(
        (key for key in saved if key.startswith("nmrdta_")),
        key=lambda key: int(key.split("_")[1]),
    )
    records = np.asarray(
        [np.exp(-1j * phases) @ np.asarray(saved[name]) for name in names]
    )
    axis = np.asarray(np.atleast_1d(experiment.dnp_var.vector), dtype=float).reshape(-1)
    averaged = np.zeros((len(axis), records.shape[1]), dtype=complex)
    for index, record in enumerate(records):
        averaged[index % len(axis)] += record
    averaged /= float(experiment.num_scans)
    return axis, averaged


def nmr_curves(root: Path, template_length: int = 128, magnitude: bool = False) -> list[tuple[float, np.ndarray, np.ndarray]]:
    _, reference_rows = _phase_cycled_nmr(root, REFERENCE_FILE)
    reference = reference_rows[0]
    start = int(np.argmax(np.abs(reference)))
    template = reference[start:start + template_length]
    denominator = float(np.vdot(template, template).real)
    curves = []
    for filename, inversion in NMR_FILES:
        axis, rows = _phase_cycled_nmr(root, filename)
        selected = rows[:, start:start + template_length]
        projection = selected @ np.conj(template) / denominator
        values = np.abs(projection) if magnitude else np.real(projection)
        curves.append((inversion, axis + inversion, values))
    return curves


def nmr_summary(root: Path, template_length: int = 128, magnitude: bool = False) -> dict:
    curves = nmr_curves(root, template_length=template_length, magnitude=magnitude)
    reference_peak = float(np.max(np.abs(curves[0][2])))
    conditions = []
    for inversion, absolute_axis, values in curves[1:]:
        expected = 2.0 * inversion
        candidates = _local_indices(absolute_axis, expected)
        detected_index = int(candidates[np.argmin(np.abs(values[candidates]))])
        exact_index = _near_index(absolute_axis, expected)
        conditions.append(
            {
                "inversion_ns": inversion,
                "expected_echo_ns": expected,
                "detected_echo_ns": float(absolute_axis[detected_index]),
                "timing_error_ns": float(abs(absolute_axis[detected_index] - expected)),
                "local_residual_fraction": float(abs(values[detected_index]) / reference_peak),
                "expected_sample_residual_fraction": float(abs(values[exact_index]) / reference_peak),
            }
        )
    inversion = np.asarray([row["inversion_ns"] for row in conditions])
    detected = np.asarray([row["detected_echo_ns"] for row in conditions])
    slope, intercept = np.polyfit(inversion, detected, 1)
    phase_440_peak = float(np.max(curves[3][2]) / np.max(curves[0][2]))
    return {
        "reference_peak": reference_peak,
        "template_length": template_length,
        "magnitude_projection": magnitude,
        "conditions": conditions,
        "max_abs_timing_error_ns": max(row["timing_error_ns"] for row in conditions),
        "short_exact_residual_fraction": conditions[0]["expected_sample_residual_fraction"],
        "phase_inversion_peak_enhancement_440": phase_440_peak,
        "detected_time_slope": float(slope),
        "detected_time_intercept_ns": float(intercept),
        "echo_condition_count": len(conditions),
    }


def summary_a(root: Path) -> dict[str, float]:
    verify_sources(root)
    result = epr_summary(root)
    return {
        "epr_max_abs_timing_error_ns": result["max_abs_timing_error_ns"],
        "epr_median_abs_timing_error_ns": result["median_abs_timing_error_ns"],
        "epr_short_echo_fraction": result["short_echo_fraction"],
        "epr_long_echo_fraction": result["long_echo_fraction"],
        "epr_echo_condition_count": float(result["echo_condition_count"]),
    }


def summary_b(root: Path) -> dict[str, float]:
    verify_sources(root)
    result = nmr_summary(root)
    return {
        "nmr_max_abs_timing_error_ns": result["max_abs_timing_error_ns"],
        "nmr_short_exact_residual_fraction": result["short_exact_residual_fraction"],
        "nmr_phase_inversion_peak_enhancement_440": result["phase_inversion_peak_enhancement_440"],
        "nmr_detected_time_slope": result["detected_time_slope"],
        "nmr_echo_condition_count": float(result["echo_condition_count"]),
    }


def perturb(card_id: str, family: str, root: Path) -> dict:
    verify_sources(root)
    if card_id == "A" and family == "definition":
        result = epr_summary(root)
        short_exact = result["conditions"][0]["expected_sample_fraction"]
        long_exact = result["conditions"][-1]["expected_sample_fraction"]
        return {
            "family": family,
            "survive": short_exact > 0.9 and long_exact < 0.55 and short_exact > long_exact,
            "values": {
                "short_expected_sample_fraction": short_exact,
                "long_expected_sample_fraction": long_exact,
                "short_to_long_ratio": short_exact / long_exact,
            },
            "note": "score the predeclared 2*t_inv sample instead of a local peak",
        }
    if card_id == "A" and family == "method":
        results = {str(length): epr_summary(root, window_length=length) for length in (192, 256, 320)}
        survives = [
            item["max_abs_timing_error_ns"] <= 88
            and item["short_echo_fraction"] > item["long_echo_fraction"]
            for item in results.values()
        ]
        return {
            "family": family,
            "survive": all(survives),
            "values": {
                length: {
                    "max_abs_timing_error_ns": item["max_abs_timing_error_ns"],
                    "short_echo_fraction": item["short_echo_fraction"],
                    "long_echo_fraction": item["long_echo_fraction"],
                }
                for length, item in results.items()
            },
            "note": "repeat author-style EPR integration with 192/256/320-point windows",
        }
    if card_id == "A" and family == "sample":
        conditions = epr_summary(root)["conditions"]
        times = np.asarray([row["inversion_ns"] for row in conditions])
        amplitudes = np.asarray([row["local_peak_fraction"] for row in conditions])
        slopes = [
            float(np.polyfit(np.log(np.delete(times, index)), np.delete(amplitudes, index), 1)[0])
            for index in range(len(times))
        ]
        return {
            "family": family,
            "survive": max(slopes) < 0,
            "values": {"leave_one_condition_slopes": slopes, "maximum_slope": max(slopes)},
            "note": "leave out each inversion-time condition; echo amplitude still decays with log time",
        }

    if card_id == "B" and family == "definition":
        signed = nmr_summary(root, magnitude=False)
        magnitude = nmr_summary(root, magnitude=True)
        return {
            "family": family,
            "survive": all(
                item["max_abs_timing_error_ns"] <= 88
                and item["short_exact_residual_fraction"] < 0.05
                and item["phase_inversion_peak_enhancement_440"] > 1.1
                for item in (signed, magnitude)
            ),
            "values": {
                "signed": {
                    "short_residual": signed["short_exact_residual_fraction"],
                    "enhancement_440": signed["phase_inversion_peak_enhancement_440"],
                },
                "magnitude": {
                    "short_residual": magnitude["short_exact_residual_fraction"],
                    "enhancement_440": magnitude["phase_inversion_peak_enhancement_440"],
                },
            },
            "note": "replace signed complex projection with projection magnitude",
        }
    if card_id == "B" and family == "method":
        results = {str(length): nmr_summary(root, template_length=length) for length in (96, 128, 160)}
        return {
            "family": family,
            "survive": all(
                item["max_abs_timing_error_ns"] <= 88
                and item["short_exact_residual_fraction"] < 0.05
                and item["phase_inversion_peak_enhancement_440"] > 1.1
                for item in results.values()
            ),
            "values": {
                length: {
                    "max_abs_timing_error_ns": item["max_abs_timing_error_ns"],
                    "short_residual": item["short_exact_residual_fraction"],
                    "enhancement_440": item["phase_inversion_peak_enhancement_440"],
                }
                for length, item in results.items()
            },
            "note": "repeat phase-cycled NMR matched projection with 96/128/160-point templates",
        }
    if card_id == "B" and family == "sample":
        conditions = nmr_summary(root)["conditions"]
        inversion = np.asarray([row["inversion_ns"] for row in conditions])
        detected = np.asarray([row["detected_echo_ns"] for row in conditions])
        slopes = [
            float(np.polyfit(np.delete(inversion, index), np.delete(detected, index), 1)[0])
            for index in range(len(inversion))
        ]
        return {
            "family": family,
            "survive": min(slopes) > 1.9 and max(slopes) < 2.1,
            "values": {"leave_one_condition_slopes": slopes, "min_slope": min(slopes), "max_slope": max(slopes)},
            "note": "leave out each inversion-time condition; detected NMR minimum still scales as 2*t_inv",
        }
    raise ValueError(f"unsupported perturbation: {card_id}/{family}")
