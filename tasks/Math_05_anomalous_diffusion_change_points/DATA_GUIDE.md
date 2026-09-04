# Data Guide: Change-Point and Regime Structure in Anonymous Trajectories

- Task ID: `Math_05_anomalous_diffusion_change_points`
- T0: `2021-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All 240 frozen length-128 anonymous trajectories and their complete ordered time index.

## Independent analysis unit

One complete trajectory; time points within a trajectory are serially correlated and trajectories are the resampling unit.

## Variables and file groups

- `anonymous_trajectories.npz`: `trajectories shape=[240, 128] dtype=float64`, `time_index shape=[128] dtype=int64`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The frozen trajectories are model-generated and do not alone establish a physical mechanism in real systems.
- Latent labels are intentionally absent; validation must use complete trajectories and contiguous time blocks.

## Distribution boundary

The article-linked trajectory generator is MIT licensed; fixed-seed scientific trajectories are bundled with latent labels withheld.
