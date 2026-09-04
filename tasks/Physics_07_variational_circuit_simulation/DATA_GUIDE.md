# Data Guide: Accuracy and Applicability Boundaries of Classical Representations for Variational Quantum Circuits

- Task ID: `Physics_07_variational_circuit_simulation`
- T0: `2020-03-30`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

The complete deterministic computational-observation projection in the frozen source boundary.

## Independent analysis unit

One small graph, one circuit layer or gate, one fixed random-seed set, and one complete state comparison.

## Variables and file groups

- `source_data/computational_observations.json`: `graph_edge_set`, `circuit_angles`, `hidden_unit_count`, `cost_layer_fidelity`, `mixing_gate_fidelity`, `compression_fidelity`, `p1_and_p2_full_state_fidelity`

## Missingness

The complete deterministic observation file is bundled.

## Warnings

- The task material first appeared after T0; do not search for the hidden target identity.
- Same-source deterministic projection is not independent-data replication.
- Do not extrapolate small-graph fidelity results to unobserved circuit sizes without evidence.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Pinned paper-era repository: Apache-2.0. Article and supplementary-material boundary: CC BY 4.0 with attribution.
