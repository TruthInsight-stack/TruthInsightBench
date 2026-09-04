# Data Guide: Cross-Graph and Cross-State Generalization of Network Dynamics Models

- Task ID: `Information_01_dynamics_generalization`
- T0: `2024-10-02`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All graph, state, persisted-model, and computational-observation files in the complete frozen source boundary.

## Independent analysis unit

One graph panel, one state distribution, one persisted model, and one paired support shift.

## Variables and file groups

- `source_data/er_n_100_p_*.npy`: `undirected_adjacency_matrix`, `node_count`, `mean_degree`
- `source_data/results/**/data.pkl`: `stored_state`, `true_dynamics_derivative`
- `source_data/results/**/neural_network.pth`: `persisted_model_parameters`
- `source_data/computational_observations.json`: `panel_relative_error`, `paired_domain_shift_error`, `test_statistics`

## Missingness

The complete frozen source boundary is bundled.

## Warnings

- The task material first appeared after T0; do not search for the hidden target identity.
- Same-source blind rediscovery is not independent-data replication.
- The result concerns one supplied persisted model unless independent model replicates are explicitly present.
- Use computational_observations.json and the NumPy graph matrices for portable analysis. The pickle and Torch checkpoint files require a compatible PyTorch runtime; report model replay only if it was actually executed.
- Pickle and checkpoint files can execute code when loaded. Treat them as non-portable source artifacts unless they are opened only inside an appropriately isolated, trusted runtime.

## Distribution boundary

The scientific payload is bundled and hash-checked.

The exact source-data record declares CC BY 4.0.
