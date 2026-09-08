# Data Guide: Compositional Invariants and Bias Boundaries in Sparse Count Systems

- Task ID: `Math_02_compositional_invariants`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen anonymous feature-count rows, sample covariates, and bias-scenario metrics within the selected composition-analysis boundary.

## Independent analysis unit

One complete sample composition or one complete simulated bias scenario; features within a composition share a closure constraint.

## Variables and file groups

- `anonymous_feature_counts.tsv`: `feature_id`, `sample_id`, `relative_count`
- `anonymous_sample_covariates.tsv`: `sample_id`, `continuous_covariate`, `coarse_group_id`
- `bias_scenario_metrics.tsv`: `scenario_id`, `condition_id`, `metric_id`, `metric_value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Relative counts are compositional and cannot be analyzed as unconstrained independent features.
- Sample- and scenario-level splits are required; random feature-row validation leaks composition identity.

## Distribution boundary

The frozen article-linked repository is MIT licensed; anonymous scientific matrices and scenario metrics are bundled with source identity isolated privately.
