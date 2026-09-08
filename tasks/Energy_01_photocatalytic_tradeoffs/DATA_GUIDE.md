# Data Guide: Cross-Modal Performance and Kinetic Tradeoffs in Photocatalytic Formulations

- Task ID: `Energy_01_photocatalytic_tradeoffs`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen numerical series for the four task modalities: product-evolution time courses, optical/action spectra, ultrafast transient traces, and cofactor-loading response.

## Independent analysis unit

One complete formulation time course, spectrum, transient trace, or loading condition; neighboring curve points are not independent replicates.

## Variables and file groups

- `conversion_timecourses.tsv`: `formulation_id`, `illumination_time_h`, `gaseous_product_umol`
- `optical_action_spectra.tsv`: `measurement_role`, `wavelength_nm`, `response_au_or_percent`
- `ultrafast_transient_traces.tsv`: `formulation_id`, `delay_ps`, `normalized_transient_signal`
- `cofactor_loading_response.tsv`: `cofactor_loading_percent`, `product_evolution_rate_umol_h`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Cross-modal association does not by itself establish a unique charge-transfer mechanism.
- The loading scan has few settings, so an apparent optimum must be treated as a bounded observation rather than a universal optimum.

## Distribution boundary

The archived target article and source-data workbook are CC BY; every numerical row in the four frozen task modalities is bundled after identity-neutral projection.
