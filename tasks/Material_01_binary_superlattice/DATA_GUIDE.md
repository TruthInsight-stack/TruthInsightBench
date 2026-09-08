# Data Guide: Low-Energy Assembly Structures of Binary Nanoparticles

- Task ID: `Material_01_binary_superlattice`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All 14 low-energy configurations in the source data and the complete accepted-local-minimum trajectory for one binary 6:6 system.

## Independent analysis unit

One global low-energy configuration for one parameter combination; frames from the same optimization trajectory are not independent replicates.

## Variables and file groups

- `structures/structure_*.xyz`: `particle_type`, `x`, `y`, `z`, `R1`, `R2`, `Chi`, `Epsilon`, `Energy`, `box`
- `structure_manifest.json`: `file`, `particle_count`, `parameters`, `sha256`
- `optimization_trajectory/LM_ENERGIES_32.txt`: `accepted_local_minimum_energy`
- `optimization_trajectory/LM_MOVIE_32.xyz`: `frame`, `particle_type`, `x`, `y`, `z`, `header_parameters`

## Missingness

No curator-imposed row deletion or subsampling is present in the frozen scientific payload.

## Warnings

- A finite cluster's particle-count ratio need not equal the stoichiometry of an infinite lattice exactly.
- Any neighbor-based result must state its cutoff and include at least one cutoff perturbation.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Article and supplementary source data: CC BY 4.0. Author-code redistribution is handled separately from the scientific payload.
