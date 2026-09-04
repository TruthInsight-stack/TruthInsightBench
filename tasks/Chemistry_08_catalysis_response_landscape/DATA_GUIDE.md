# Data Guide: Response Structure and Transfer Boundaries in a Catalytic Landscape

- Task ID: `Chemistry_08_catalysis_response_landscape`
- T0: `2023-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All four frozen anonymous numerical observation sets representing a large reaction-condition landscape and multiple cross-measurement panels.

## Independent analysis unit

One complete reaction-condition observation or one complete comparison series; columns and related observations from a shared panel are not independent experiments.

## Variables and file groups

- `observation_set_1.tsv`: `observation_index`, `numeric_feature_1`, `numeric_feature_2`, `numeric_feature_3`, `numeric_feature_4`, `numeric_feature_5`, `numeric_feature_6`, `numeric_feature_7`
- `observation_set_2.tsv`: `observation_index`, `numeric_feature_1`, `numeric_feature_2`, `numeric_feature_3`
- `observation_set_3.tsv`: `observation_index`, `numeric_feature_1`, `numeric_feature_2`, `numeric_feature_3`, `numeric_feature_4`, `numeric_feature_5`, `numeric_feature_6`, `numeric_feature_7`, `numeric_feature_8`, `numeric_feature_9`, `numeric_feature_10`, `numeric_feature_11`, `numeric_feature_12`, `numeric_feature_13`, `numeric_feature_14`, `numeric_feature_15`, `numeric_feature_16`, `numeric_feature_17`, `numeric_feature_18`, `numeric_feature_19`, `numeric_feature_20`, `numeric_feature_21`
- `observation_set_4.tsv`: `observation_index`, `numeric_feature_1`, `numeric_feature_2`, `numeric_feature_3`, `numeric_feature_4`, `numeric_feature_5`, `numeric_feature_6`, `numeric_feature_7`, `numeric_feature_8`, `numeric_feature_9`, `numeric_feature_10`, `numeric_feature_11`, `numeric_feature_12`, `numeric_feature_13`, `numeric_feature_14`, `numeric_feature_15`, `numeric_feature_16`, `numeric_feature_17`, `numeric_feature_18`, `numeric_feature_19`, `numeric_feature_20`, `numeric_feature_21`, `numeric_feature_22`, `numeric_feature_23`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Anonymous numeric features limit chemical-mechanism interpretation; claims must remain at the measured response-landscape level.
- Whole-condition or whole-series holdouts are required to avoid local interpolation leakage.

## Distribution boundary

The selected source-data workbook is part of a CC BY Nature Communications record; all selected anonymous numeric rows are bundled.
