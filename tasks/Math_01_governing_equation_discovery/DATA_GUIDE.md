# Data Guide: Governing-Equation Structure and Failure Boundaries in Sparse Spatiotemporal Fields

- Task ID: `Math_01_governing_equation_discovery`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

Both declared scientific files are included in the repository and are ready after checkout.

## Scientific scope

Both complete frozen spatiotemporal arrays: one dense synthetic field with its full time and space grids and one sparse three-channel experimental field with its full grids.

## Independent analysis unit

One complete time slice, spatial block, or experimental channel; neighboring grid points are correlated samples of the same field.

## Variables and file groups

- `coordinate_definitions.json`: `schema: "truthinsightbench-coordinate-definitions-v1"`, `task_id: "Math_01_governing_equation_discovery"`, `synthetic_field: {"field": "dimensionless simulated response", "space_coordinate": "dimensionless simulation position", "time_coordinate": "dimensionless simulation time"}`, `experimental_field: {"channel_role": "three identically prepared replicate profiles", "field": "cell-density profile in source measurement units", "physical_space_unit": "not encoded in the retained author MAT array; do not invent a unit", "retained_values": "25 through 1875 in 50-unit increments", "source_domain": "0 through 1900 with zero-flux boundary coordinates", "space_coordinate": "source assay coordinate", "time_coordinate": "hours", "time_values": [0, 12, 24, 36, 48]}`
- `spatiotemporal_fields.npz`: `synthetic_time shape=[101] dtype=float64`, `synthetic_space shape=[201] dtype=float64`, `synthetic_field shape=[101, 201] dtype=float64`, `experimental_time shape=[5] dtype=uint8`, `experimental_space shape=[38] dtype=uint16`, `experimental_field shape=[5, 38, 3] dtype=float64`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Numerical differentiation and boundary handling can create spurious governing terms and must be varied.
- A good in-sample derivative fit is insufficient; candidate equations require contiguous-time or initial-condition validation.
- The experimental coordinates represent hours, source assay position, and cell density as declared in coordinate_definitions.json; do not invent a physical space unit absent from the author MAT array.

## Distribution boundary

The processed numerical payload is not covered by the repository-level license; see the provenance record for source information.
