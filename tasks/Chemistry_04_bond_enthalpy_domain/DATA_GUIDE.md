# Data Guide: Structural Domains and Model Boundaries of Organic Bond Dissociation Enthalpies

- Task ID: `Chemistry_04_bond_enthalpy_domain`
- T0: `2019-10-18`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

The complete bond-dissociation-enthalpy database, experimental source tables, and a deterministic 543-observation benchmark view.

## Independent analysis unit

Count raw rows, unique molecules, and unique molecule-fragment pairs separately; each experimental bond observation is an error-analysis unit.

## Variables and file groups

- `normalized_views/experimental_benchmark.tsv`: `observation_id`, `parent_structure_smiles`, `fragment_a_smiles`, `fragment_b_smiles`, `experimental_bde_kcal_mol`, `quantum_estimate_bde_kcal_mol`, `learned_estimate_bde_kcal_mol`, `reference_estimate_bde_kcal_mol`
- `source_data/bde_db_full.csv.gz`: `complete row-level database`, `molecule`, `fragment`, `bond_type`, `bond_enthalpy`
- `source_data/experimental_source_tables.xlsx`: `complete workbook sheets and numeric payload`

## Missingness

The complete frozen database and experimental-source boundary is present.

## Warnings

- Raw rows, unique molecules, and unique bond-dissociation observations are distinct statistical units.
- Experimental references come from heterogeneous literature and are not independent repeats from one laboratory.

## Distribution boundary

The scientific payload is bundled and hash-checked.

Complete database: MIT. Article supplementary data: CC BY 4.0.
