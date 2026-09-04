# TruthInsightBench Generic Agent Launcher

Execute the scientific discovery task in the current workspace now. Do not wait for confirmation and do not stop after writing a plan.

1. Read `PublicTaskContract.md`, `ResearchObjective.md`, `DATA_GUIDE.md`, `data_manifest.json`, and `validate_output.py` completely. Inventory every file under `data/`, then inspect its schema and content with code or format-appropriate tools. Cover every scientifically relevant file group, but do not paste large data files into the conversation merely to satisfy a literal full-file read.
2. Treat the current task and data as the only benchmark-provided scientific input for this run. Do not read another task, a historical run, another Agent's deliverables, scoring material, acquisition provenance, or private evaluator assets. External literature search is optional and is governed by the public contract's `T0` boundary.
3. Workspace data are the only direct evidence for core claims, and `data/` is the complete task dataset. Do not download, substitute, or merge external scientific data or precomputed task results. Eligible external sources first public strictly before `T0` may be used only for background, methods, or alternative explanations. For every source used, record a citation and stable URL or identifier plus its first-public date. Do not use search to identify or reconstruct the hidden source behind the task.
4. Run the analysis and save reproducible artifacts. A plan, suggested code, or model self-report is not evidence that execution occurred.
5. Put every scored deliverable under `output/` while retaining the Agent's native organization. Make the root Markdown report the complete and only authoritative scientific account. Clearly label 2 to 5 registered findings and link each one to resolvable analysis and result evidence.
6. Use the platform's normal end-to-end scientific workflow and preserve its native process files. No benchmark-specific process tree is required beyond the five semantic delivery roles. Before finishing, run `python3 validate_output.py output` and repair every semantic delivery error without changing data-warranted conclusions. If a tool or method fails, retain the real error and continue with an executable alternative.

Do not cater to a presumed answer. Report support, refutation, null effects, conflicts, boundaries, and unresolved outcomes exactly as the data warrant.

# Generic Launch

Start the current task. Do not finish until the core metadata have been read, the complete data inventory has been inspected, the analysis has actually run, all minimum deliverables have been created, and `python3 validate_output.py output` passes.
