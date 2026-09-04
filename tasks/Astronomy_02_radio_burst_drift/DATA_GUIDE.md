# Data Guide: Drift Distributions and Conditional Differences in Discrete Radio Bursts

- Task ID: `Astronomy_02_radio_burst_drift`
- T0: `2022-10-13`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

A losslessly deidentified copy of all 47 source-data sheets, plus deterministic views of five complete dynamic spectra, six directional count series, and twelve drift-population distributions.

## Independent analysis unit

One complete dynamic-spectrum event or complete source-polarization series; pixels and adjacent directional coordinates are nested repeats.

## Variables and file groups

- `normalized_views/drift_population_series.tsv`: `population_series_id`, `anonymous_source_group`, `polarization_channel`, `drift_regime`, `distribution_bin_center`, `event_count`
- `normalized_views/dynamic_spectrum_event_metadata.tsv`: `event_id`, `time_min_s`, `time_max_s`, `frequency_min_MHz`, `frequency_max_MHz`, `reported_event_drift_rate_MHz_s`, `frequency_bins`, `time_bins`
- `normalized_views/event_*_dynamic_spectrum.tsv`: `frequency_MHz`, `ordered time columns in seconds`
- `normalized_views/orientation_count_series.tsv`: `orientation_series_id`, `relative_coordinate`, `event_count`
- `source_data/all_source_data_deidentified.xlsx`: `complete workbook sheets and numeric payload`, `sheet relationships retained under anonymous identifiers`

## Missingness

The complete frozen source-data boundary is present.

## Warnings

- There are only five complete events; pixel count is not an event-level sample size.
- Drift-population composition can establish conditional association, not a causal acceleration mechanism or source.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Article and source data: CC BY 4.0.
