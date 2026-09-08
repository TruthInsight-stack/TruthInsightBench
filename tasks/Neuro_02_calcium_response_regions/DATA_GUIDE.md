# Data Guide: Region-, Condition-, and Batch-Dependent Calcium Response Structure

- Task ID: `Neuro_02_calcium_response_regions`
- T0: `2020-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen per-cell calcium response summaries across anonymous anatomical regions, conditions, recording batches, slices, and ages.

## Independent analysis unit

One cell nested within a recording slice, batch, and biological source; cells from a shared slice are not independent animals.

## Variables and file groups

- `cellular_calcium_response_summaries.tsv`: `observation_id`, `recording_batch_id`, `animal_age_days`, `slice_id`, `condition`, `anatomical_region_id`, `mean_amplitude_dff0`, `mean_time_to_peak_s`, `mean_peak_width_s`, `mean_peak_prominence_dff0`, `mean_peak_frequency_hz`, `mean_auc_au`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Cell-level differences cannot substitute for biological-source or recording-batch replication.
- Region, condition, age, slice, and batch effects may be confounded and require grouped checks.

## Distribution boundary

The selected source-data workbook is CC BY and all retained per-cell measurements are bundled after region and recording identity removal.
