# Data Guide: Temperature-Dependent Response Boundaries in Interface Catalysis

- Task ID: `Energy_06_interface_response_boundaries`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen anonymous condition-response curves and the independent condition-check curves across the temperature-like setting.

## Independent analysis unit

One complete condition curve; adjacent temperature settings are ordered scan points rather than independent systems.

## Variables and file groups

- `anonymous_condition_response_curves.tsv`: `condition_id`, `temperature_like_setting`, `rate_like_response`
- `independent_condition_check_curves.tsv`: `condition_id`, `temperature_like_setting`, `rate_like_response`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The curves support operating relationships but not a unique microscopic interface mechanism.
- Whole-condition and contiguous-setting holdouts are required; random curve-point splitting is invalid.

## Distribution boundary

The selected curves are from a CC BY article-linked source-data archive and are bundled under anonymous condition identifiers.
