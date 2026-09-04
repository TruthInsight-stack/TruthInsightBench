# Data Guide: Frequency- and Time-Domain Boundaries of Controlled-Dynamics Forecasts

- Task ID: `Physics_03_controlled_dynamics_forecast`
- T0: `2021-04-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

A losslessly deidentified copy of all 65 source-data sheets, plus three frequency scans, three complete trajectories, and one auxiliary anonymous numeric table.

## Independent analysis unit

One complete ordered control scan or complete trajectory; adjacent control points and time points are correlated observations.

## Variables and file groups

- `normalized_views/anonymous_observations.tsv`: `observation_index`, `numeric_feature_1`, `numeric_feature_2`, `numeric_feature_3`
- `normalized_views/control_a_frequency_response.tsv`: `control_a`, `reference_frequency`, `forecast_frequency`
- `normalized_views/control_b_frequency_response.tsv`: `control_b`, `reference_frequency`, `forecast_frequency`
- `normalized_views/signed_control_frequency_response.tsv`: `signed_control`, `reference_frequency`, `forecast_frequency`
- `normalized_views/trajectory_condition_*.tsv`: `time`, `drive`, `reference_response`, `forecast_response`
- `source_data/all_source_data_deidentified.xlsx`: `complete workbook sheets and numeric payload`

## Missingness

The complete frozen source-data boundary is present.

## Warnings

- Reference and forecast channels are paired outputs, not independent experiments.
- Good frequency prediction does not by itself establish full-trajectory accuracy or a correct microscopic mechanism.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Article and source data: CC BY 4.0. Author analysis code: MIT.
