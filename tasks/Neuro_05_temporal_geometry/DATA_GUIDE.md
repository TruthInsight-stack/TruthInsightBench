# Data Guide: Event-State Timing and Unit-Distance Geometry

- Task ID: `Neuro_05_temporal_geometry`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen event-level records from two recording sets, unit-distance latency summaries, and the complete multichannel event panel.

## Independent analysis unit

One complete recording set, unit, or multichannel event; adjacent events within a recording are repeated observations.

## Variables and file groups

- `event_recording_set_a.tsv`: `event_index`, `event_relative_time`, `pre_event_signal`
- `event_recording_set_b.tsv`: `event_index`, `event_relative_time`, `pre_event_signal`
- `unit_distance_mean_latency.tsv`: `unit_index`, `pair_distance`, `mean_event_latency`
- `unit_distance_latency_variability.tsv`: `unit_index`, `pair_distance`, `event_latency_sd`
- `multichannel_event_panel.tsv`: `event_index`, `unit_index`, `pre_event_signal`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Anonymous states and units permit timing and geometry claims but not a named cellular or synaptic mechanism.
- Evidence must be separated across recording-, event-, unit-, and panel-level analysis units.

## Distribution boundary

The selected source-data sheets are from a CC BY article record and are bundled after recording and neural identities are removed.
