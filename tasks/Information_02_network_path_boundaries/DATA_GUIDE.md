# Data Guide: Path-Length and Efficiency Boundaries in Directed Networks

- Task ID: `Information_02_network_path_boundaries`
- T0: `2018-10-29`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All reference networks, example networks, deterministic observations, and generator inputs in the complete frozen source boundary.

## Independent analysis unit

One node count N, one arc count L, and one directed graph construction.

## Variables and file groups

- `source_data/Figs/directed_sparse_reference.net`: `sparse_directed_reference_graph`
- `source_data/Examples/Data/Dir_*.net`: `author_supplied_directed_network_examples`
- `source_data/computational_observations.json`: `node_count`, `arc_count`, `average_path_length`, `global_efficiency`, `diameter`, `density`, `structural_checks`
- `source_data/**`: `complete generator and reference-network source boundary`

## Missingness

The complete frozen source boundary is bundled.

## Warnings

- The task material first appeared after T0; do not search for the hidden target identity.
- Same-source blind rediscovery is not independent-data replication.
- State connectivity assumptions explicitly; path length and efficiency have different behavior for unreachable pairs.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Pinned author code: Apache-2.0. Open article and supplementary-material boundary: CC BY 4.0 with attribution.
