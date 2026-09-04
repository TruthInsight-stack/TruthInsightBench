# Data Guide: Electron and Nuclear Spin Dynamics under Phase Inversion

- Task ID: `Physics_01_dynamic_nuclear_polarization_echoes`
- T0: `2024-06-26`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All 13 raw MAT experiment files in the frozen data record are included. Author processing scripts are outside the scientific payload.

## Independent analysis unit

One complete scan under a fixed phase-inversion condition per MAT file; time points within a scan are correlated repeated measurements.

## Variables and file groups

- `raw/*dnp_echo*.mat`: `EPR raw traces`, `indirect time axis`, `inversion and detection metadata`
- `raw/*dnpdata_dnp_contact*.mat`: `eight-step phase cycles`, `indirect time axis`, `NMR direct time axis`
- `raw/*dnpdata_nmr_only*.mat`: `NMR reference scan for phase and normalization`
- `raw/*.mat`: `complete no-inversion and phase-inversion EPR/NMR scans`

## Missingness

All files in the frozen scientific-data record are present.

## Warnings

- Times and pulse metadata in filenames are not observations; process the raw waveforms.
- EPR requires signal parsing, down-conversion, echo-window selection, and phase correction; NMR requires eight-step phase cycling, reference phase, and normalization.
- Time points within a scan are correlated and must not inflate confidence as independent samples.
- Validate processing-window, phase-reference, and normalization choices with prespecified alternatives.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Article and raw data/processing scripts in the frozen data record: CC BY 4.0.
