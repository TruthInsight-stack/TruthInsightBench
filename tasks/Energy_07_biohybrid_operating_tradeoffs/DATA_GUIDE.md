# Data Guide: Dose, Time, and Operating Tradeoffs in a Biohybrid Energy System

- Task ID: `Energy_07_biohybrid_operating_tradeoffs`
- T0: `2023-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen time-response curves, perturbation summaries, and dose-response measurements within the selected operating boundary.

## Independent analysis unit

One complete condition-specific trajectory or one complete dose setting with its uncertainty; fitted columns are supporting representations, not independent observations.

## Variables and file groups

- `dose_response.tsv`: `dose`, `response_mean`, `response_sd`
- `perturbation_time_response.tsv`: `time`, `control_mean`, `control_sd`, `condition_a_mean`, `condition_a_sd`, `condition_b_mean`, `condition_b_sd`
- `time_response_curves.tsv`: `time`, `promoter_like_signal`, `condition_a`, `condition_b`, `condition_c`, `fit_a`, `fit_b`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The dose scan contains few settings, so a universal optimum cannot be inferred.
- Fitted and raw time-response columns share measurements and must not be double-counted as independent evidence.

## Distribution boundary

The selected source-data measurements derive from a CC BY Nature Communications record and are bundled after identity-neutral projection.
