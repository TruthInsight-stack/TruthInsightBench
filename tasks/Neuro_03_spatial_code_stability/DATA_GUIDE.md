# Data Guide: Perturbation Boundaries of Spatial-Code Activity and Stability

- Task ID: `Neuro_03_spatial_code_stability`
- T0: `2021-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen unit-session activity and spatial metrics plus tracked-unit repeated- and new-environment stability summaries.

## Independent analysis unit

One animal with its units and sessions, or one tracked unit within an animal; units from one animal are nested observations.

## Variables and file groups

- `unit_session_metrics.tsv`: `observation_id`, `animal_id`, `condition`, `session_phase`, `average_rate_hz`, `out_of_field_rate_hz`, `in_field_rate_hz`, `spatial_regularity_score`, `information_rate`, `spacing`, `orientation`, `speed_modulation_score`
- `tracked_unit_spatial_stability.tsv`: `tracked_unit_id`, `animal_id`, `condition`, `repeated_environment_stability_1`, `repeated_environment_stability_2`, `new_environment_stability_1`, `new_environment_stability_2`, `new_environment_stability_3`, `between_repeated_environment_stability`, `repeated_environment_change`, `novel_vs_repeated_change`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The number of animals is limited; unit counts do not increase the independent animal sample size.
- The task provides summary metrics rather than raw spike trains and cannot reconstruct every spatial field definition.

## Distribution boundary

The selected CC BY source-data measurements are bundled with animal, unit, and experimental identities anonymized.
