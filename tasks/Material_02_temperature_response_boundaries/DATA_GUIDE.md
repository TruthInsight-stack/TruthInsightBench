# Data Guide: Temperature-Dependent Magnetic and Electrical Response Boundaries

- Task ID: `Material_02_temperature_response_boundaries`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen magnetic field-scan rows at three temperatures and both frozen electrical sequence measurements for one anonymous oxide sample source.

## Independent analysis unit

One complete field scan or electrical pulse sequence; neighboring points within a scan are ordered repeated measurements.

## Variables and file groups

- `bulk_magnetization_300K.tsv`: `time_s`, `temperature_K`, `field_Oe`, `magnetic_moment_emu`, `moment_std_err_emu`
- `bulk_magnetization_860K.tsv`: `time_s`, `temperature_K`, `field_Oe`, `magnetic_moment_emu`, `moment_std_err_emu`
- `bulk_magnetization_870K.tsv`: `time_s`, `temperature_K`, `field_Oe`, `magnetic_moment_emu`, `moment_std_err_emu`
- `electric_switching_corrected.tsv`: `sequence_coordinate`, `polarization_like_signal`
- `electric_current_voltage.tsv`: `sequence_coordinate`, `current_or_voltage_signal`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Only one sample source is represented, so cross-material generalization is unsupported.
- Background subtraction, branch segmentation, and sequence direction can materially change loop-like summaries and must be perturbed.

## Distribution boundary

The complete archived Zenodo task source is CC BY 4.0; the task bundles all numerical rows used by the frozen magnetic and electrical measurements.
