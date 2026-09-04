# Data Guide: Unit- and Site-Level Environmental Transfer Boundaries

- Task ID: `Life_08_environmental_transfer_boundary`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen unit-level, site-level, symptom-linked, and quantitative-assay observations within the selected environmental sampling boundary.

## Independent analysis unit

One sampled unit with its sites and study days; sites from the same unit are repeated observations rather than independent units.

## Variables and file groups

- `unit_level_observations.tsv`: `unit_id`, `illness_day`, `stay_day`, `quantitative_assay_value`, `any_environment_positive`
- `site_level_observations.tsv`: `unit_id`, `site_role_id`, `positive`
- `symptom_assay_surface_summary.tsv`: `unit_id`, `illness_day`, `symptom_present`, `surface_positive_percent`, `quantitative_assay_value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The observational data cannot by themselves prove a transmission route or direction.
- Unit-level blocking is required to avoid treating many sampled sites as independent subjects.

## Distribution boundary

The selected observations are from a CC BY article record and are bundled after unit and site identity removal.
