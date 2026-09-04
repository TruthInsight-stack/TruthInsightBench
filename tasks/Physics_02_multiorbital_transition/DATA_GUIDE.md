# Data Guide: Change Boundaries of Multiple Observables under a Control Parameter

- Task ID: `Physics_02_multiorbital_transition`
- T0: `2023-04-21`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

Five complete numerical figure-data archives and deterministic long-table views; no control point or observable branch was removed.

## Independent analysis unit

One complete control-parameter branch or independent gap scan; spectral entries at one control point are nested observations.

## Variables and file groups

- `normalized_views/control_vs_correlation.tsv`: `control_ratio`, `secondary_interaction_ratio`, `inverse_correlation_length`
- `normalized_views/control_vs_gap_case_A.tsv`: `control_ratio`, `gap_like_observable`
- `normalized_views/control_vs_gap_case_B.tsv`: `control_ratio`, `gap_like_observable`
- `normalized_views/control_vs_local_observables.tsv`: `control_ratio`, `total_magnetic_moment`, `charge_fluctuation`
- `normalized_views/control_vs_topological_proxy.tsv`: `control_ratio`, `spectrum_value`, `degeneracy`
- `source_data/all_source_data_deidentified.zip`: `complete deidentified archive members`, `preserved numeric payload and member structure`

## Missingness

No control-parameter point or observable branch was removed.

## Warnings

- These are finite-size numerical-model data, not an experimental material.
- Describe any boundary as a finite-sampling crossover or transition interval; a numerical zero crossing alone does not establish a strict phase transition.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Scientific data: CC BY 4.0.
