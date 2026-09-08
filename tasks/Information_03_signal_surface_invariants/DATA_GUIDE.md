# Data Guide: Robust Regions and Failure Boundaries in Replicated Input-Output Surfaces

- Task ID: `Information_03_signal_surface_invariants`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen replicated response-surface rows, the complete independent response panel, and its supplied independent fit-error matrix.

## Independent analysis unit

One complete input-grid condition with its replicate structure; adjacent grid cells and repeated measurements are not independent systems.

## Variables and file groups

- `replicated_response_panel.tsv`: `input_axis_1_index`, `input_axis_2_index`, `signal_bin_index`, `output_channel_index`, `replicate_id`, `response_value`
- `independent_response_panel.tsv`: `input_axis_1_index`, `input_axis_2_index`, `signal_bin_index`, `output_channel_index`, `replicate_id`, `response_value`
- `independent_fit_error.tsv`: `input_axis_index`, `replicate_id`, `fit_error`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Normalization may create apparent invariants and must be perturbed.
- Grid interpolation cannot substitute for validation on complete held-out rows, columns, or the independent panel.

## Distribution boundary

The archived article and nested source-data matrices are CC BY; task tables retain the complete frozen panels under anonymous axes and construct identifiers.
