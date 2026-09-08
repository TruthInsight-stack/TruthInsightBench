# Data Guide: Assortative Structure and Spatial Boundaries in Annotated Networks

- Task ID: `Information_04_connectome_assortative_structure`
- T0: `2023-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen node summaries, three annotation profiles, and trimming-sensitivity measurements for the anonymous network systems.

## Independent analysis unit

One network system or complete node profile; nodes within a network and repeated annotations are dependent observations.

## Variables and file groups

- `network_node_summary.tsv`: `region_index`, `mean_homophily`, `node_strength`, `mean_connection_distance`
- `network_profile_a.tsv`: `region_index`, `coord_x`, `coord_y`, `coord_z`, `annotation_1`, `annotation_2`, `annotation_3`, `annotation_4`, `annotation_5`
- `network_profile_b.tsv`: `region_index`, `coord_x`, `coord_y`, `coord_z`, `annotation_1`
- `network_profile_c.tsv`: `region_index`, `coord_x`, `coord_y`, `coord_z`, `annotation_1`, `annotation_2`, `annotation_3`
- `trimming_sensitivity.tsv`: `system_index`, `annotation_index`, `result_kind`, `trim_percent_1`, `trim_percent_2`, `trim_percent_3`, `trim_percent_4`, `trim_percent_5`, `trim_percent_6`, `trim_percent_7`, `trim_percent_8`, `trim_percent_9`, `trim_percent_10`, `trim_percent_11`, `trim_percent_12`, `trim_percent_13`, `trim_percent_14`, `trim_percent_15`, `trim_percent_16`, `trim_percent_17`, `trim_percent_18`, `trim_percent_19`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Node-level associations do not establish developmental or causal mechanisms.
- Inference must respect network-level clustering and test trimming and spatial-distance sensitivity.

## Distribution boundary

The source-data tables are from a CC BY article archive and are bundled with system and anatomical identities removed.
