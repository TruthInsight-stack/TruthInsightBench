# Data Guide: Long-Record Sulfate Event Structure in an Ice Core

- Task ID: `EarthScience_04_ice_core_event_structure`
- T0: `2019-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

The complete frozen PANGAEA numerical record of depth, age, sulfate concentration, density, and event-signal annotations.

## Independent analysis unit

One ordered core interval; neighboring intervals are serially correlated and must be validated with contiguous blocks or event-level units.

## Variables and file groups

- `observations.tsv`: `c01_depth_top_m`, `c02_depth_ice_snow_m`, `c03_samp_thick_cm`, `c04_sulfate_ug_kg`, `c05_density_ice_g_cm_3`, `c06_depth_w_e_m`, `c07_signal`, `c08_age_a_ad_ce`, `c09_age_ka_bp`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Age, depth, and density encode related coordinates and must not be counted as independent confirmations.
- Peak detection depends on background and window definitions, which must be reported and perturbed.

## Distribution boundary

The complete frozen PANGAEA record is CC BY 4.0 and is projected without row deletion, imputation, or reordering; named event labels are deterministically pseudonymized.
