# Data Guide: Wavelength- and Temperature-Dependent Spectral Response Boundaries

- Task ID: `Astronomy_05_stellar_spectral_boundary`
- T0: `2023-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen photon-flux grid rows across three effective-temperature slices, wavelength, and an ordered composition index, plus the independent normalized spectral series.

## Independent analysis unit

One complete temperature-by-wavelength spectrum or one complete ordered response series; neighboring wavelengths and composition settings are correlated scan points.

## Variables and file groups

- `normalized_spectral_response.tsv`: `wavelength_nm`, `response_level_1`, `response_level_2`, `response_level_3`, `response_level_4`, `response_level_5`
- `photon_flux_grid.tsv`: `effective_temperature_K`, `wavelength_nm`, `metallicity_index`, `photon_flux_cm2`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- The observations are radiation and atmosphere calculations and cannot alone establish biological or evolutionary causality.
- Spectral smoothness makes random-point validation invalid; hold out contiguous wavelength bands or complete temperature slices.

## Distribution boundary

The complete source tables derive from a CC BY article-linked source-data archive and are bundled as identity-neutral long-form tables.
