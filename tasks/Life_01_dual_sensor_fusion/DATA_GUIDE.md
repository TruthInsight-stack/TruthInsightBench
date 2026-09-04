# Data Guide: Sensor Conditions and Fusion Phenotypes in a Single-Vesicle System

- Task ID: `Life_01_dual_sensor_fusion`
- T0: `2024-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All 15 source-data tables and five raw or representative images embedded in the source workbook.

## Independent analysis unit

One complete run identified in a source table; time bins and workbook mean columns within a run are not independent replicates.

## Variables and file groups

- `source_tables/source_table_01.csv`: `condition`, `independent_run`, `total_fusion_percent`, `fusion_in_first_two_frames_percent`
- `source_tables/source_table_02.csv`: `condition`, `independent_run`, `survival`, `calcium_triggered_fusion`, `kinetics`
- `source_tables/source_table_03.csv`: `condition`, `independent_run`, `total_fusion_percent`, `early_fusion_percent`
- `source_tables/source_table_04.csv`: `condition`, `independent_run`, `rescue_total_fusion`, `rescue_early_fusion`
- `source_tables/source_table_05.csv`: `competitor_concentration`, `independent_run`, `bound_target_concentration`
- `source_tables/source_table_06.csv`: `time`, `calcium_concentration`, `sensor_model_output`, `fusion_kinetics`
- `source_tables/source_table_07.csv`: `basal_calcium_condition`, `independent_run`, `fusion_amount`, `fusion_timing`
- `source_tables/source_table_08.csv`: `membrane_topology_condition`, `independent_run`, `fusion_response`
- `source_tables/source_table_09.csv`: `condition`, `independent_run`, `docked_vesicles`, `clamped_vesicles`
- `source_tables/source_table_10.csv`: `calcium_green_control`, `fluorescence_control`, `transmission_control`
- `source_tables/source_table_11.csv`: `binding_condition`, `independent_run`, `bound_concentration`
- `source_tables/source_table_12.csv`: `time`, `calcium_input`, `sensor_response`
- `source_tables/source_table_13.csv`: `clamp_configuration`, `time`, `model_output`
- `source_tables/source_table_14.csv`: `basal_calcium_condition`, `docking`, `clamping`, `fusion`
- `source_tables/source_table_15.csv`: `membrane_condition`, `TargetComplex_mobility`, `fusion_response`
- `derived_for_navigation/fusion_dose_independent_runs.csv`: `condition`, `metric`, `run`, `value_pct`
- `derived_for_navigation/binding_competition_runs.csv`: `competitor_concentration_uM`, `run`, `bound_target_uM`

## Missingness

No curator-imposed row deletion or table subsampling is present.

## Warnings

- Recompute from run-level columns; workbook mean columns are not independent evidence.
- Keep total fusion, first-two-frame fusion, survival percentage, and per-bin frequency distinct.
- Public labels such as Sensor A and Sensor B are anonymous; do not reverse-identify the source from values, figure numbers, or molecular names.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Article and source data: CC BY 4.0. Two associated simulation repositories: GPL-3.0.
