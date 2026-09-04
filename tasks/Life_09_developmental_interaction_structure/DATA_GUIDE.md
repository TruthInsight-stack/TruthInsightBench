# Data Guide: Stage-Indexed Interaction Structure Across Development

- Task ID: `Life_09_developmental_interaction_structure`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

The complete frozen stage-indexed interaction table, including anonymous interaction and cell-pair identifiers, association values, and interaction-strength-like measurements.

## Independent analysis unit

One interaction-cell-pair combination followed across ordered stages; repeated stages for the same pair are dependent.

## Variables and file groups

- `stage_indexed_interactions.tsv`: `stage_index`, `interaction_id`, `cell_pair_id`, `association_pvalue`, `interaction_strength_like_value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Association strength and p-values do not uniquely identify direct biological communication or causality.
- Interaction or cell-pair families must be held out intact when testing stage generalization.

## Distribution boundary

The complete selected source-data table is CC BY and is bundled with biological identities removed while retaining ordered stage and numerical structure.
