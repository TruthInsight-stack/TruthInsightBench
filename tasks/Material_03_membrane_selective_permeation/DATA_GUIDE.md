# Data Guide: Device-Scale and Local Transport Heterogeneity in Two-Dimensional Membranes

- Task ID: `Material_03_membrane_selective_permeation`
- T0: `2023-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen device time responses, condition responses, bias-current curves, conductivity replicates, local current observations, and the complete spatial current map from the three archived source-data workbooks.

## Independent analysis unit

One device, one complete ordered bias-current curve, one conductivity specimen, or one contiguous spatial block. Each class-level local-current vector is a single source-provided descriptive distribution because no parent device/region identifier is available for its trace values.

## Variables and file groups

- `bias_current_curves.tsv`: `sample_class`, `bias_V`, `current_A`
- `condition_response.tsv`: `test_condition`, `device_id`, `time_day`, `dimensionless_delta`
- `conductivity_replicates.tsv`: `sample_class`, `replicate_id`, `proton_conductivity_mS_cm2`
- `device_time_response.tsv`: `device_id`, `time_day`, `dimensionless_delta`
- `local_current_distributions.tsv`: `sample_class`, `observation_id`, `current_pA`
- `spatial_current_map.tsv`: `pixel_id`, `x_um`, `y_um`, `current_pA`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Spatial-map pixels require block-aware inference; unparented local-current trace values support descriptive sensitivity analysis only and cannot be bootstrapped as independent specimens.
- Device-scale and local measurements may constrain one another but do not uniquely identify a microscopic transport path.

## Distribution boundary

All three archived source-data workbooks are CC BY 4.0 and every task-relevant numerical sheet is projected losslessly under anonymous labels.
