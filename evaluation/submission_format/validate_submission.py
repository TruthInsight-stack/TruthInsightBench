#!/usr/bin/env python3
"""Validate the semantic TruthInsightBench main-run output contract.

The benchmark standardizes scientific functions, not Agent-native directory
layouts.  ``Result.md`` is the sole authority for the registered findings.
Supporting analysis, evidence, and reproduction material may be supplied by
direct paths or by a resolvable native artifact/task registry.  The validator
checks structural completion and traceability only; it does not score science.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".f",
    ".f90",
    ".f95",
    ".go",
    ".h",
    ".hpp",
    ".ipynb",
    ".java",
    ".jl",
    ".js",
    ".m",
    ".py",
    ".r",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".ts",
}

TABLE_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".h5",
    ".hdf5",
    ".json",
    ".jsonl",
    ".mat",
    ".npy",
    ".npz",
    ".parquet",
    ".tsv",
    ".xls",
    ".xlsx",
}

FIGURE_SUFFIXES = {
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}

# Runtime and workflow utilities are provenance, not scientific analysis code.
CONTROL_CODE_NAMES = {
    "approval_manager.py",
    "bootstrap_runtime.py",
    "cell_validator.py",
    "dashboard_manager.py",
    "domain_plugin_selector.py",
    "file_registry_manager.py",
    "final_pipeline_audit.py",
    "flowsearch_manager.py",
    "input_capability_selector.py",
    "literature_task_recovery.py",
    "notebook_manager.py",
    "post_report_qa.py",
    "post_task_check.py",
    "recovery_state_inspector.py",
    "report_asset_manager.py",
    "subagent_log_archiver.py",
    "task_contracts.py",
    "task_index_manager.py",
    "validate_output.py",
    "validate_result.py",
    "world_model_manager.py",
}

# These machine-readable files describe orchestration rather than scientific
# results.  A scientific ``results.json`` remains eligible.
PROCESS_DATA_NAMES = {
    "approval_request.json",
    "approval_response.json",
    "approval_state.json",
    "campaign.json",
    "dashboard.json",
    "dashboard_state.json",
    "data_manifest.json",
    "decision_log.jsonl",
    "file_registry.json",
    "flowsearch_plan.json",
    "input_capability_plan.json",
    "plugin_profile.json",
    "planning_thread.jsonl",
    "report_asset_manifest.json",
    "run_manifest.json",
    "run_state.json",
    "runtime_manifest.json",
    "task_index.json",
    "task_progress.json",
    "validation_report.json",
}

REPRODUCTION_HEADING = re.compile(
    r"^#{2,6}\s+(?:reproducibility|reproduction|how\s+to\s+reproduce|"
    r"reproducing\s+(?:the\s+)?analysis)\b",
    re.IGNORECASE | re.MULTILINE,
)
METHODS_HEADING = re.compile(
    r"^#{2,6}\s+(?:methods?|methodology|analysis\s+workflow|computational\s+workflow)\b",
    re.IGNORECASE | re.MULTILINE,
)
FINDING_PATTERNS = (
    re.compile(r"^(?:discovery|finding)\s*(?:d\s*)?([1-9][0-9]*)\b", re.IGNORECASE),
    re.compile(r"^d([1-9][0-9]*)\s*(?:[\s:—–-]|$)", re.IGNORECASE),
)
ARTIFACT_ID = re.compile(r"\bF[0-9]{3,4}\b", re.IGNORECASE)
TASK_ID = re.compile(r"\bT(?:ASK[\s_-]*)?0*([1-9][0-9]{0,2})\b", re.IGNORECASE)
NOTEBOOK_CELL = re.compile(r"\bnotebook\s+cells?\b", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_row(path: Path, output: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _root_aliases(output: Path, canonical_name: str) -> list[Path]:
    if not output.is_dir():
        return []
    return sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name.casefold() == canonical_name.casefold()
    )


def _select_unique_content(
    paths: list[Path],
    *,
    label: str,
    preferred_name: str,
    errors: list[str],
    warnings: list[str],
) -> Path | None:
    nonempty = [path for path in paths if path.stat().st_size > 0]
    if not nonempty:
        errors.append(f"missing_or_empty_{label}")
        return None
    by_hash: dict[str, list[Path]] = {}
    for path in nonempty:
        by_hash.setdefault(sha256(path), []).append(path)
    if len(by_hash) > 1:
        errors.append(
            f"ambiguous_{label}_with_different_content:"
            + ",".join(path.name for path in nonempty)
        )
        return None
    if len(nonempty) > 1:
        warnings.append(
            f"duplicate_{label}_aliases_with_identical_content:"
            + ",".join(path.name for path in nonempty)
        )
    preferred = next((path for path in nonempty if path.name == preferred_name), None)
    return preferred or nonempty[0]


def _finding_markers(report_text: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(report_text.splitlines(), start=1):
        stripped = raw_line.strip()
        heading_match = re.match(r"^(#{2,6})\s+", stripped)
        is_heading = bool(heading_match)
        is_bold_label = stripped.startswith("**")
        if not (is_heading or is_bold_label):
            continue
        label = re.sub(r"^#{2,6}\s+", "", stripped)
        label = label.lstrip("*").strip()
        for pattern in FINDING_PATTERNS:
            match = pattern.match(label)
            if match:
                markers.append(
                    {
                        "finding_number": int(match.group(1)),
                        "line": line_number,
                        "marker": raw_line.strip(),
                        "heading_level": len(heading_match.group(1)) if heading_match else None,
                    }
                )
                break
    return markers


def _finding_sections(report_text: str, markers: list[dict[str, Any]]) -> dict[int, str]:
    lines = report_text.splitlines()
    sections: dict[int, str] = {}
    for index, marker in enumerate(markers):
        start = marker["line"] - 1
        end = markers[index + 1]["line"] - 1 if index + 1 < len(markers) else len(lines)
        heading_level = marker.get("heading_level")
        if heading_level is not None:
            for line_index in range(start + 1, end):
                heading = re.match(r"^(#{1,6})\s+", lines[line_index].strip())
                if heading and len(heading.group(1)) <= heading_level:
                    end = line_index
                    break
        number = marker["finding_number"]
        sections[number] = sections.get(number, "") + "\n" + "\n".join(lines[start:end])
    return sections


def _is_analysis_source(path: Path, output: Path) -> bool:
    if path.suffix.casefold() not in CODE_SUFFIXES:
        return False
    relative = path.relative_to(output)
    if "__pycache__" in relative.parts or "protocols" in relative.parts:
        return False
    return path.name.casefold() not in CONTROL_CODE_NAMES


def _notebook_execution(path: Path) -> dict[str, Any]:
    result = {"path": str(path), "parseable": False, "has_code": False, "has_execution": False}
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return result
    result["parseable"] = True
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        result["has_code"] = True
        if cell.get("execution_count") is not None or cell.get("outputs"):
            result["has_execution"] = True
    return result


def _is_result_artifact(path: Path, output: Path) -> bool:
    relative = path.relative_to(output)
    if (
        "__pycache__" in relative.parts
        or path.name.startswith("_")
        or path.name.casefold() in PROCESS_DATA_NAMES
    ):
        return False
    suffix = path.suffix.casefold()
    if suffix in FIGURE_SUFFIXES:
        return not (relative.parent == Path(".") and path.stem.casefold() == "result")
    if suffix not in TABLE_SUFFIXES:
        return False
    name = path.name.casefold()
    return not (
        name.endswith("_state.json")
        or name.endswith("_manifest.json")
        or name.endswith("_registry.json")
        or name.endswith("_receipt.json")
        or name.endswith("_report.json")
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _is_directly_referenced(path: Path, output: Path, text: str, basename_counts: dict[str, int]) -> bool:
    relative = path.relative_to(output).as_posix()
    basename = path.name
    if relative in text or f"output/{relative}" in text:
        return True
    return basename_counts.get(basename.casefold(), 0) == 1 and basename in text


def _normalize_registry_path(output: Path, raw: str) -> Path | None:
    candidate_text = raw.strip().replace("\\", "/")
    if not candidate_text:
        return None
    markers = ("/output/", "output/")
    relative: str | None = None
    for marker in markers:
        if marker in candidate_text:
            relative = candidate_text.split(marker, 1)[1]
            break
    if relative is None and not candidate_text.startswith("/"):
        relative = candidate_text
    if relative is None:
        return None
    candidate = (output / relative).resolve()
    try:
        candidate.relative_to(output)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _artifact_registry(output: Path) -> tuple[dict[str, Path], list[Path], list[str]]:
    registry: dict[str, Path] = {}
    sources: list[Path] = []
    warnings: list[str] = []
    for path in sorted(output.rglob("file_registry.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            warnings.append(f"unreadable_native_registry:{path.relative_to(output).as_posix()}")
            continue
        sources.append(path)
        rows = payload.get("files", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            target = None
            for key in ("relative_path", "path"):
                if isinstance(row.get(key), str):
                    target = _normalize_registry_path(output, row[key])
                    if target is not None:
                        break
            if target is not None:
                registry[str(row["id"]).upper()] = target
    return registry, sources, warnings


def _task_directories(output: Path) -> dict[int, list[Path]]:
    result: dict[int, list[Path]] = {}
    for path in output.rglob("*"):
        if not path.is_dir():
            continue
        match = re.match(r"^task[_-]0*([1-9][0-9]{0,2})(?:[_-]|$)", path.name, re.IGNORECASE)
        if match:
            result.setdefault(int(match.group(1)), []).append(path)
    return result


def _scope_for_section(
    section: str,
    *,
    output: Path,
    all_candidates: list[Path],
    basename_counts: dict[str, int],
    registry: dict[str, Path],
    task_directories: dict[int, list[Path]],
) -> tuple[set[Path], list[str]]:
    support: set[Path] = {
        path
        for path in all_candidates
        if _is_directly_referenced(path, output, section, basename_counts)
    }
    resolved_tokens: list[str] = []
    scoped_directories: set[Path] = set()
    for token in sorted({token.upper() for token in ARTIFACT_ID.findall(section)}):
        target = registry.get(token)
        if target is not None:
            support.add(target)
            scoped_directories.add(target.parent)
            resolved_tokens.append(token)
    for raw_number in TASK_ID.findall(section):
        number = int(raw_number)
        for directory in task_directories.get(number, []):
            scoped_directories.add(directory)
            resolved_tokens.append(f"T{number:02d}")
    # A native reference such as ``F002 + notebook cell 5`` resolves through
    # the registered task report to the notebook in that same task directory.
    if NOTEBOOK_CELL.search(section):
        for directory in scoped_directories:
            support.update(directory.rglob("*.ipynb"))
    for directory in scoped_directories:
        support.update(path for path in all_candidates if directory in path.parents)
    return support, sorted(set(resolved_tokens))


def validate_output(output: Path) -> dict[str, Any]:
    """Return a deterministic semantic-validation receipt for one output tree."""

    if output.is_symlink():
        absolute = output.absolute()
        return {
            "schema": "truthinsightbench-semantic-output-validation",
            "passed": False,
            "errors": ["unsafe_output_entry:symbolic_link:."],
            "warnings": [],
            "output": str(absolute),
        }
    output = output.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not output.is_dir():
        errors.append("missing_output_directory")
        return {
            "schema": "truthinsightbench-semantic-output-validation",
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "output": str(output),
        }

    unsafe_entries: list[str] = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if path.is_symlink():
            unsafe_entries.append(f"symbolic_link:{relative}")
        elif not path.is_file() and not path.is_dir():
            unsafe_entries.append(f"non_regular_file:{relative}")
    if unsafe_entries:
        errors.extend(f"unsafe_output_entry:{item}" for item in unsafe_entries)
        return {
            "schema": "truthinsightbench-semantic-output-validation",
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "output": str(output),
        }

    report = _select_unique_content(
        _root_aliases(output, "Result.md"),
        label="authoritative_markdown_report",
        preferred_name="Result.md",
        errors=errors,
        warnings=warnings,
    )
    report_text = ""
    if report is not None:
        report_text = _read_text(report)
        if len(report_text.strip()) < 200:
            errors.append("authoritative_report_too_short_for_structural_review")
        if "REPLACE_" in report_text:
            errors.append("authoritative_report_contains_unreplaced_marker")

    markers = _finding_markers(report_text)
    finding_numbers = sorted({item["finding_number"] for item in markers})
    if not 2 <= len(finding_numbers) <= 5:
        errors.append(f"finding_count_must_be_2_to_5:detected={len(finding_numbers)}")
    elif finding_numbers != list(range(1, len(finding_numbers) + 1)):
        errors.append(
            "finding_numbers_must_be_consecutive_from_1:"
            + ",".join(map(str, finding_numbers))
        )

    files = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.stat().st_size > 0
    ]
    analysis_sources = [path for path in files if _is_analysis_source(path, output)]
    result_artifacts = [path for path in files if _is_result_artifact(path, output)]
    if not analysis_sources:
        errors.append("missing_scientific_code_or_notebook")
    if not result_artifacts:
        errors.append("missing_machine_readable_result_or_supporting_figure")

    notebook_execution = [
        _notebook_execution(path)
        for path in analysis_sources
        if path.suffix.casefold() == ".ipynb"
    ]
    executed_notebooks = {
        Path(item["path"]).resolve()
        for item in notebook_execution
        if item["has_code"] and item["has_execution"]
    }

    reproduction_files = _root_aliases(output, "reproduce.md")
    reproduction = (
        _select_unique_content(
            reproduction_files,
            label="reproduction_document",
            preferred_name="reproduce.md",
            errors=errors,
            warnings=warnings,
        )
        if reproduction_files
        else None
    )
    reproduction_mode: str | None = None
    if reproduction is not None:
        reproduction_mode = "companion_markdown"
    elif report is not None and REPRODUCTION_HEADING.search(report_text):
        reproduction_mode = "section_in_authoritative_report"
    elif report is not None and METHODS_HEADING.search(report_text) and analysis_sources:
        reproduction_mode = "methods_section_plus_analysis_sources"
    elif executed_notebooks:
        reproduction_mode = "executed_native_notebook_trace"
    else:
        errors.append("missing_reproducible_method_or_executed_notebook_trace")

    registry, registry_sources, registry_warnings = _artifact_registry(output)
    warnings.extend(registry_warnings)
    task_directories = _task_directories(output)
    candidate_support = analysis_sources + result_artifacts + [
        path
        for path in files
        if path.suffix.casefold() in {".md", ".html"} and path != report
    ]
    basename_counts: dict[str, int] = {}
    for path in candidate_support:
        name = path.name.casefold()
        basename_counts[name] = basename_counts.get(name, 0) + 1

    finding_sections = _finding_sections(report_text, markers)
    finding_traceability: list[dict[str, Any]] = []
    all_resolved_support: set[Path] = set()
    for number in finding_numbers:
        support, tokens = _scope_for_section(
            finding_sections.get(number, ""),
            output=output,
            all_candidates=candidate_support,
            basename_counts=basename_counts,
            registry=registry,
            task_directories=task_directories,
        )
        source_support = sorted(path for path in support if path in analysis_sources)
        artifact_support = sorted(path for path in support if path in result_artifacts)
        notebook_evidence = sorted(path for path in source_support if path in executed_notebooks)
        has_analysis = bool(source_support)
        has_result_evidence = bool(artifact_support or notebook_evidence)
        if not has_analysis:
            errors.append(f"finding_{number}_has_no_resolvable_analysis_source")
        if not has_result_evidence:
            errors.append(f"finding_{number}_has_no_resolvable_result_evidence")
        all_resolved_support.update(support)
        finding_traceability.append(
            {
                "finding_number": number,
                "resolved_native_tokens": tokens,
                "analysis_sources": [
                    path.relative_to(output).as_posix() for path in source_support
                ],
                "result_evidence": [
                    path.relative_to(output).as_posix() for path in artifact_support
                ],
                "executed_notebook_evidence": [
                    path.relative_to(output).as_posix() for path in notebook_evidence
                ],
                "passed": has_analysis and has_result_evidence,
            }
        )

    report_row = _file_row(report, output) if report is not None else None
    reproduction_row = _file_row(reproduction, output) if reproduction is not None else None
    return {
        "schema": "truthinsightbench-semantic-output-validation",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "output": str(output),
        "authoritative_report": report_row,
        "registered_findings": {
            "count": len(finding_numbers),
            "numbers": finding_numbers,
            "markers": markers,
            "traceability": finding_traceability,
        },
        "reproduction": {
            "mode": reproduction_mode,
            "document": reproduction_row,
        },
        "analysis_sources": [_file_row(path, output) for path in analysis_sources],
        "executed_notebooks": [
            _file_row(path, output) for path in sorted(executed_notebooks)
        ],
        "result_artifacts": [_file_row(path, output) for path in result_artifacts],
        "native_evidence_registry": {
            "sources": [_file_row(path, output) for path in registry_sources],
            "resolved_ids": {
                key: value.relative_to(output).as_posix()
                for key, value in sorted(registry.items())
            },
        },
        "resolved_support_files": [
            _file_row(path, output) for path in sorted(all_resolved_support)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Agent output directory")
    parser.add_argument("--receipt", type=Path, help="Optional receipt destination")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = validate_output(args.output)
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
