# TruthInsightBench Public Scientific Discovery Contract

## Objective

Use the scientific data in the current task workspace as the sole direct evidence for core findings. Independently propose, test, and report reproducible, bounded, and falsifiable scientific findings. External information that satisfies the time boundary below may support only background, methods, or alternative explanations.

The task does not presume that any candidate relationship is true. Null effects, reverse relationships, conditional conflicts, applicability boundaries, and evidence-backed unresolved results are valid conclusions. Never assume that a relationship must exist merely because it falls within the research objective.

## Scientific inputs visible to the Agent

The benchmark supplies only the canonical scientific content below. A native launcher may expose the components as separate files or embed them losslessly in the platform's required single research-goal entry file. That presentation difference adds no scientific direction or evidence. A platform's native system prompt, tools, and execution framework belong to the evaluated Agent and are not benchmark hints.

1. This public task contract.
2. The current task's neutral research objective.
3. The task-level data interpretation notes in `DATA_GUIDE.md`.
4. The data inventory, fields, units, analysis unit, missingness notes, license boundary, and hashes in `data_manifest.json`.
5. The current task's scientific data under `data/`.
6. The public semantic completion validator `validate_output.py`. It verifies delivery roles and traceability only and is not scientific evidence.

Evaluator-private materials, scoring assets, other task workspaces, historical runs, historical scores, and other Agents' outputs are outside the scientific input boundary and must not be sought or used.

## Reporting language

Write the authoritative report and Agent-authored scientific narrative in English. Preserve raw data values, symbols, source labels, code identifiers, and file names when translation would alter the evidence or break reproducibility.

## Evidence and time boundary

- Workspace data are the only direct evidence for a core scientific claim.
- External literature search is optional. When a task specifies `T0`, an external source may be used only if it was first public strictly before `T0`, and only for background, methods, or alternative explanations.
- For every external source used, record a citation and stable URL or identifier plus its first-public date in the report or a linked source log. If that date cannot be verified as strictly before `T0`, do not use the source.
- Do not use external search to identify or reconstruct the hidden source behind the task, its authors, title, identifier, repository, exact wording, target values, or conclusions first released on or after `T0`.
- Do not download, substitute, or merge an external scientific dataset or precomputed task result into the evidentiary analysis. The workspace `data/` directory is the complete dataset for this task.
- Do not describe an unrun method, unread datum, unobserved result, or background reference as direct evidence from this task.

## Required research actions

1. Identify files, variables, units, independent analysis units, repeated-measure structure, and missing-data boundaries.
2. For each key candidate relationship, compare at least one null or simple baseline explanation and discuss additional alternatives where the data allow.
3. Run code that produces the core numerical evidence. File names, comments, visual impressions, and model self-reports are not executed evidence.
4. Apply at least one task-appropriate robustness check to each submitted finding, such as complete-unit or batch leave-out, an alternative definition, threshold sensitivity, an alternative model, or an outlier-treatment comparison.
5. Distinguish direct observations, data-supported inferences, mechanism interpretations, and unresolved questions.
6. Report failed, conflicting, or unresolved tests that materially bear on a submitted claim; do not silently discard them.

## Finding registration

Register 2 to 5 high-level research conclusions for scoring. A conclusion may report a supported relationship, reverse relationship, null effect, conditional conflict, applicability boundary, or evidence-backed unresolved result. The minimum of two does not mean two affirmative claims. Do not split one conclusion into synonyms or lower the evidence standard to meet the minimum.

Make the registered set mechanically identifiable in the final report. Label each registered conclusion consecutively from 1 using a Markdown heading or bold label that begins with `Finding 1`, `Discovery 1`, or `D1` (case-insensitive), followed by the corresponding labels through the final finding. The label is only an index; the scientific content remains ordinary report prose.

For every registered finding, state at least:

- object and scope;
- relationship, direction, or boundary;
- core quantitative result and analysis unit;
- supporting analysis and result evidence, identified either by exact path or by a native artifact/task identifier that resolves to real files;
- robustness result;
- counterevidence, limitation, and alternative explanations; and
- an executable new observation or independent replication that could falsify or validate it.

Supporting observations, background, limitations, and evidence rows for one finding do not automatically become additional findings.

## Semantic delivery contract

The benchmark standardizes the scientific functions of the output, not a shared internal directory layout. Every run must provide all five semantic roles below under `output/`:

1. **Authoritative findings:** one non-empty root Markdown report, matched case-insensitively as `Result.md`, containing the complete set of 2 to 5 registered findings, qualifications, limitations, and validation proposals.
2. **Executed analysis source:** scientific code or notebooks actually used. These may appear anywhere in the Agent's native output tree.
3. **Inspectable result evidence:** at least one machine-readable result artifact or supporting figure. It may appear anywhere in the native output tree; no `tables/` or `figures/` directory name is prescribed.
4. **Reproduction trace:** either a companion reproduction document, a clearly headed reproduction or methods section linked to the analysis source, or an executed native notebook trace with source and outputs.
5. **Finding-to-evidence traceability:** every registered finding must resolve to real analysis and result evidence. Direct paths and unambiguous file names are valid. Native artifact IDs, task IDs, notebook-cell references, and artifact registries are equally valid when they resolve to real files.

The root Markdown report is the sole authority for which findings are submitted for scoring. Supporting task reports, notebooks, registries, dashboards, HTML, PDF, logs, and other native process files provide evidence and provenance but cannot introduce a scored claim that is absent from the root report.

The benchmark does not request a second Agent-authored structured restatement of the report. After the run, the evaluator generates a read-only semantic inventory that maps the five roles to real files and hashes without rewriting the Agent's conclusions.

Preserve the platform's normal scientific-process and state files. No additional standardized process tree is required beyond the five semantic roles. Unreferenced file count, report length, and native-process complexity do not increase the scientific score.

## Integrity and failure handling

Do not fabricate commands, numbers, code execution, figures, references, network access, or error recovery. If one library, file, fit, or tool fails, preserve the real error and continue with an executable simpler method. If full completion is impossible, submit the completed artifacts and state the exact gap.
