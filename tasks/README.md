# Task format

Tasks are grouped by domain and numbered contiguously from `01` within each
domain. The registry orders the 40 tasks by domain and then by task number.

Every task has the same layout:

```text
<Domain>_<NN>_<slug>/
├── ResearchObjective.md
├── DATA_GUIDE.md
├── data_manifest.json
└── data/
```

- `ResearchObjective.md` gives the neutral research objective.
- `DATA_GUIDE.md` explains scope, analysis units, variables, missingness, reuse
  boundaries, and task-specific warnings.
- `data_manifest.json` is the machine-readable inventory. All manifests use the
  same field structure and describe every bundled file by path, role, size, and
  SHA-256.
- `data/` contains the scientific payload. Formats such as TSV, CSV, JSON, NPZ,
  MAT, XLSX, and domain-specific text are retained where appropriate; format
  differences reflect the scientific modality rather than different task
  contracts.
