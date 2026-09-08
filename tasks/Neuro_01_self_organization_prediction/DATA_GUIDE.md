# Data Guide: Self-Organization and Prediction Boundaries in Multi-Session Neural Networks

- Task ID: `Neuro_01_self_organization_prediction`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All six response MAT files under the frozen source repository's in-vitro data boundary.

## Independent analysis unit

One complete trial or experiment in a MAT cell array; 25,600 time points and sessions within a trial are not independent replicates.

## Variables and file groups

- `raw/response_data_*.mat`: `s: hidden sources (time, 2)`, `o: sensory stimulus (time, 32)`, `r: neural response (time, 64)`
- `source_shapes.json`: `condition`, `independent_trials`, `trial_shape`, `field_shapes`

## Missingness

The frozen data contain no curator-imposed row deletion or subsampling.

## Warnings

- Time points are not independent replicates; do not report n=25,600 as the scientific sample size.
- Model selection, population filtering, and standardization must be explicit in executable code.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Frozen source repository: GPL-3.0; the data boundary is archived with its upstream license.
