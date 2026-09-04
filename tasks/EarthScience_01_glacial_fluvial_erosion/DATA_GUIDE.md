# Data Guide: Scale and Measurement Boundaries of Glacial and Nonglacial Erosion Rates

- Task ID: `EarthScience_01_glacial_fluvial_erosion`
- T0: `2024-10-17`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

The complete public compilation, including main tables, topographic and lithologic auxiliary tables, and earlier compilations.

## Independent analysis unit

One published erosion-rate estimate per row; multiple rows from the same paper, site, or compilation are not fully independent replicates.

## Variables and file groups

- `raw/glacial_erosion_Earth.tsv`: `Erosion rate (mm/yr)`, `Time interval (yr)`, `Methodology`, `Type`, `Area (km2)`, `Precipitation (mm/yr)`, `Slope (m/km)`, `Latitude`
- `raw/nonglacial_erosion_Earth.tsv`: `Erosion rate (mm/yr)`, `Time interval (yr)`, `Methodology`, `Type`, `Area (km2)`, `Precipitation (mm/yr)`, `Slope (m/km)`, `Latitude`
- `raw/**`: `complete auxiliary and prior-compilation fields retained from the frozen source`

## Missingness

Missing values and extremes are retained and must be audited at row level.

## Warnings

- Rates span orders of magnitude; define averaging and regression in log space and report filtering rules.
- Rows sharing a source, site, or compilation may be correlated.
- A rate-timescale slope can arise mechanically from fixed measured thickness; stratify by Methodology.
- Any exclusion, winsorization, or weighting must retain a row-level audit trail.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Article and Zenodo code/data record: CC BY 4.0.
