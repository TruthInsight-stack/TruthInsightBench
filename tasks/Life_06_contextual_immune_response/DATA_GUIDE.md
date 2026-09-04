# Data Guide: Context-Dependent Immune Response Across Compartments and Interventions

- Task ID: `Life_06_contextual_immune_response`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen compartment, intervention, multi-condition, and time-resolved response replicates in the selected experimental boundary.

## Independent analysis unit

One biological replicate nested within an assay, compartment, condition, or time course; repeated endpoints from a shared assay are not independent experiments.

## Variables and file groups

- `multi_condition_response_replicates.tsv`: `response_class_id`, `compound_setting_id`, `replicate_id`, `response_value`
- `compartment_response_replicates.tsv`: `assay_id`, `compartment_id`, `endpoint_type`, `condition_role`, `replicate_id`, `response_value`
- `mechanistic_intervention_replicates.tsv`: `assay_id`, `intervention_setting_id`, `replicate_id`, `response_value`
- `time_resolved_response.tsv`: `condition_role`, `time_units`, `replicate_id`, `normalized_response_percent`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Compartment and intervention associations do not by themselves identify a unique cellular mechanism.
- Validation must hold out complete assays or condition groups rather than random replicate rows.

## Distribution boundary

The selected measurements are from a CC BY article and source-data archive and are bundled with biological identities replaced by neutral roles.
