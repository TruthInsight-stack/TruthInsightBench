# Data Guide: Time-Resolved Gravity Residuals and Candidate Spectral Structure

- Task ID: `Astronomy_04_gravity_spectral_structure`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All 22 frozen complete time-resolved passes, the ranked full spectral-parameter grid, both supporting candidate-grid slices, the mode-frequency rows, and the uniform-alternative comparison rows for the anonymous gravitating system.

## Independent analysis unit

One complete time-resolved pass or one complete candidate spectral parameter combination; adjacent time samples and neighboring grid rows are not independent replicates.

## Variables and file groups

- `candidate_spectrum_comparison.tsv`: `candidate_family`, `max_velocity_cm_s`, `min_velocity_cm_s`, `peak_frequency_microhz`, `peak_width_microhz`, `information_criterion`
- `mode_frequency_grid.tsv`: `angular_degree`, `radial_order`, `mode_frequency_microhz`, `surface_velocity_cm_s`, `coupling_coefficient`
- `pass_residual_traces.tsv`: `pass_id`, `relative_time_min`, `residual_model_A_mm_s`, `residual_model_C_mm_s`, `residual_model_B_mm_s`
- `ranked_full_parameter_grid.tsv`: `candidate_rank`, `max_velocity_cm_s`, `min_velocity_cm_s`, `peak_frequency_microhz`, `frequency_width_microhz`, `delta_information_criterion`
- `uniform_alternative_comparison.tsv`: `uniform_velocity_cm_s`, `information_criterion`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The number of complete passes is limited, so claims must use blocked-time or whole-pass checks rather than treating time samples as replicates.
- A lower information criterion supports a relative comparison within the supplied candidate family, not proof that the family is physically unique.

## Distribution boundary

Target article and archived source-data tables are CC BY; task tables are lossless, identity-neutral projections of the frozen numerical source tables.
