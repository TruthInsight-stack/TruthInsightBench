# Data Guide: Energy- and Angle-Dependent Molecular Photoionization Dynamics

- Task ID: `Chemistry_07_photoionization_delay`
- T0: `2021-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

Complete deterministic long-format TSV projections of all datasets in the four frozen HDF5 scientific files, covering experimental and theoretical energy-angle observations, dipole-matrix elements, angular coefficients, and phase/delay curves.

## Independent analysis unit

One measured dipole-matrix energy block or one complete theory calculation block. The dense experiment-like phase/delay grid is a spline-derived representation of 13 measured energy blocks, so its cells are correlated derived values rather than independent observations.

## Variables and file groups

- `angle_energy_observations_datasets.tsv`: `dataset_path`, `index_0`, `index_1`, `value`
- `angular_coefficients_datasets.tsv`: `dataset_path`, `index_0`, `index_1`, `value`
- `coordinate_definitions.json`: `schema: "truthinsightbench-coordinate-definitions-v1"`, `task_id: "Chemistry_07_photoionization_delay"`, `angle_coordinate: {"dataset_axis": "index_1 for energy-angle phase, delay, and magnitude arrays", "definition": "emission_angle_deg = index_1", "range_deg": [0, 180], "step_deg": 1}`, `phase_delay_conversion: {"formula": "delay_as = d(unwrapped_phase_rad)/d(photon_energy_eV) * hbar_as_eV", "hbar_as_eV": 658.2119569}`, `experimental_grid_provenance: {"dense_phase_delay_grid_status": "derived_spline_grid", "independence_warning": "Dense derived grid cells are not independent experimental measurements.", "measured_dipole_matrix_energy_blocks": 13, "measured_energy_values_eV": [22.5, 23.25, 24.8, 26.35, 27.9, 29.45, 31.0, 32.55, 34.1, 35.65, 37.2, 38.75, 48.4]}`
- `dipole_matrix_elements_datasets.tsv`: `dataset_path`, `index_0`, `index_1`, `value`
- `phase_and_delay_curves_datasets.tsv`: `dataset_path`, `index_0`, `index_1`, `value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Interpolation between experimental and theoretical energy grids must be declared and perturbed.
- Large angular variation near a local energy region is not by itself evidence for a unique microscopic mechanism.
- The public coordinate metadata defines emission angle and phase-to-delay conversion; dense derived grid cells must not be counted as experimental replicates.

## Distribution boundary

Zenodo scientific data: CC BY 4.0. Bundled TSV files are deterministic projections of the cited source files.
