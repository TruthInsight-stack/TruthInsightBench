# Data Guide: State- and Composition-Dependent Long-Wavelength Emission Structure

- Task ID: `Material_06_nir_emission_structure`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen aggregate-state and dilute-state emission spectra plus long-wavelength response measurements across design series and composition fractions.

## Independent analysis unit

One complete design-series spectrum or composition series; neighboring wavelengths and compositions are ordered, correlated scan points.

## Variables and file groups

- `aggregate_state_emission_spectra.tsv`: `design_series`, `aggregate_state_id`, `composition_fraction`, `wavelength_nm`, `relative_emission_au`
- `dilute_state_emission_spectra.tsv`: `design_series`, `wavelength_nm`, `relative_emission_au`
- `long_wavelength_response.tsv`: `design_series`, `composition_fraction`, `long_wavelength_response_au`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Spectral intensity alone does not uniquely identify an emission mechanism.
- Complete spectra, design series, or wavelength bands must be held out instead of random points.

## Distribution boundary

The complete selected source-data spectra are from a CC BY article archive and are bundled after compound and figure identity removal.
