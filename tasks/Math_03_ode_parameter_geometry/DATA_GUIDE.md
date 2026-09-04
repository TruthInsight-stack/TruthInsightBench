# Data Guide: Parameter Geometry and Identifiability Boundaries in ODE Fits

- Task ID: `Math_03_ode_parameter_geometry`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen parameter trajectories, singular-value spectra, and paired measurement-simulation values across solver families and progress settings.

## Independent analysis unit

One complete parameter trajectory, solver family, spectrum, or paired observation series; progress points within a run are correlated.

## Variables and file groups

- `parameter_trajectories.tsv`: `parameter_index`, `solver_family_id`, `progress_index`, `normalised_distance_like_value`
- `singular_value_spectrum.tsv`: `mode_index`, `log_singular_value`
- `measurement_simulation_pairs.tsv`: `observation_index`, `measurement_like_value`, `simulation_like_value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Small measurement-simulation error does not prove parameter identifiability.
- Whole-trajectory and solver-family holdouts are required to avoid progress-point leakage.

## Distribution boundary

The complete selected source-data archive is CC BY and is bundled under neutral parameter and solver identifiers.
