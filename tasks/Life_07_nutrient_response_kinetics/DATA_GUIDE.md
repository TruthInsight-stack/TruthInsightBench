# Data Guide: Nutrient-Response Kinetics Across System States and Tissues

- Task ID: `Life_07_nutrient_response_kinetics`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen baseline phenotypes, circulating-input measurements, nutrient-response time courses, and tissue-uptake measurements.

## Independent analysis unit

One biological replicate within a system state, tissue, input condition, or time course; repeated measures sharing a state are nested.

## Variables and file groups

- `nutrient_response_timecourse.tsv`: `system_state_id`, `time_min`, `replicate_id`, `normalized_signaling_response`
- `circulating_input_measurement.tsv`: `input_condition_role`, `system_state_id`, `replicate_id`, `measured_abundance_au`
- `tissue_uptake_measurement.tsv`: `tissue_id`, `system_state_id`, `replicate_id`, `uptake_measurement_au`
- `baseline_system_phenotype.tsv`: `anatomical_measure_id`, `system_state_id`, `replicate_id`, `measurement_value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Cross-panel association does not uniquely establish molecular causality.
- System-state and tissue groups must be held out intact when testing generalization.

## Distribution boundary

The selected source-data workbook is CC BY and is bundled with tissue and system identities represented by neutral identifiers.
