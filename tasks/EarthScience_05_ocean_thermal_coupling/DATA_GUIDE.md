# Data Guide: Coupling, Phase, and Transition Structure in a Continuous Ocean Record

- Task ID: `EarthScience_05_ocean_thermal_coupling`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

The complete frozen sediment-depth, dual-age-coordinate, and isotope-proxy record, with every ordered row retained.

## Independent analysis unit

One contiguous depth or age interval; adjacent core rows are serially correlated and must not be treated as independent replicates.

## Variables and file groups

- `observations.tsv`: `c01_depth_sed_m_merged`, `c02_age_ka_bp_1_see_reference_s`, `c03_age_ka_bp_2_see_reference_s`, `c04_c_wuellerstorfi_18o_pdb_vs_vpdb_analytical_error_0`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- One continuous record cannot by itself establish a global Earth-system causal mechanism.
- Alternative age coordinates, smoothing windows, and contiguous block choices must be compared explicitly.

## Distribution boundary

The complete PANGAEA numerical table is CC BY 4.0 and is bundled with normalized, ordinal column names.
