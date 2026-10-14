"""Capability contracts for bulk transcriptome analysis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITIES: dict[str, dict[str, Any]] = {
    "inspect_count_matrix": {
        "name": "inspect_count_matrix",
        "aliases": ["count matrix qc", "inspect counts", "表达矩阵检查", "count表检查"],
        "biological_intent": "Validate gene-by-sample raw count matrix structure before analysis.",
        "description": "Inspect count matrix shape, sample columns, integer-like counts, and group prefixes.",
        "tool": "bio_count_matrix_inspect",
        "parameters": {"file_path": "string"},
        "requires": {"files": ["count_matrix"]},
        "produces": {"state": ["count_matrix_inspected", "sample_columns", "sample_groups"]},
        "side_effects": ["read count matrix only"],
        "return_structure": {"sample_count": "int", "samples": "list[str]", "total_counts_seen": "dict"},
        "failure_modes": ["missing_file", "malformed_table", "non_numeric_count"],
    },
    "infer_sample_groups": {
        "name": "infer_sample_groups",
        "aliases": ["group inference", "sample groups", "组织分组识别"],
        "biological_intent": "Infer biological groups from sample naming conventions such as el1/el2/el3.",
        "description": "Infer group prefixes and replicate counts from sample columns.",
        "tool": "transcriptome_state_inspect",
        "parameters": {"count_matrix_path": "string", "target_group": "string=optional"},
        "requires": {"state": ["sample_columns"]},
        "produces": {"state": ["sample_groups", "replicate_counts"]},
        "side_effects": ["none"],
        "return_structure": {"groups": "dict[str,int]", "target_replicates": "int"},
        "failure_modes": ["ambiguous_sample_names", "insufficient_replicates"],
    },
    "check_deseq2_environment": {
        "name": "check_deseq2_environment",
        "aliases": ["DESeq2 env", "R environment", "差异分析环境检查"],
        "biological_intent": "Ensure the DESeq2 runtime is available before differential expression.",
        "description": "Check Rscript and DESeq2 package loadability.",
        "tool": "workflow_preflight_check",
        "parameters": {"analysis_type": "deseq2"},
        "requires": {"state": ["count_matrix_inspected"]},
        "produces": {"state": ["deseq2_environment_checked"]},
        "side_effects": ["runs Rscript environment probe"],
        "return_structure": {"rscript": "string", "deseq2_loads": "bool"},
        "failure_modes": ["rscript_missing", "deseq2_missing", "r_package_version_conflict"],
    },
    "check_omicverse_bulk_backend": {
        "name": "check_omicverse_bulk_backend",
        "aliases": ["OmicVerse backend", "omicverse bulk", "OmicVerse环境检查"],
        "biological_intent": "Ensure OmicVerse bulk transcriptome methods are available before using them.",
        "description": "Check importability and required bulk symbols: pyDEG, deseq2_normalize, Matrix_ID_mapping.",
        "tool": "transcriptome_omicverse_check",
        "parameters": {},
        "requires": {"state": ["count_matrix_inspected"]},
        "produces": {"state": ["omicverse_backend_checked"]},
        "side_effects": ["imports omicverse if installed"],
        "return_structure": {"available": "bool", "version": "string", "missing_symbols": "list[str]"},
        "failure_modes": ["backend_missing", "backend_import_error", "missing_bulk_symbols"],
    },
    "omicverse_pydge_tissue_vs_rest": {
        "name": "omicverse_pydge_tissue_vs_rest",
        "aliases": ["OmicVerse DEG", "pyDEG", "bulk differential expression", "OmicVerse差异表达"],
        "biological_intent": "Run bulk RNA-seq differential expression through OmicVerse pyDEG.",
        "description": "Use ov.bulk.pyDEG to run target-vs-rest differential expression with DESeq2/edgeR/limma-style methods.",
        "tool": "transcriptome_omicverse_deg",
        "parameters": {
            "count_matrix_path": "string",
            "target_group": "string",
            "method": "DEseq2|edger|limma|ttest",
            "alpha": "number=0.05",
        },
        "requires": {
            "state": [
                "count_matrix_inspected",
                "sample_groups",
                "target_has_replicates",
                "rest_has_replicates",
                "omicverse_backend_checked",
            ]
        },
        "produces": {
            "files": ["omicverse_full_results_csv", "omicverse_significant_csv", "omicverse_summary_json"],
            "state": ["differential_expression_completed"],
        },
        "side_effects": ["imports omicverse", "writes result CSV and summary JSON"],
        "return_structure": {"status": "completed|backend_missing|error", "summary": "dict"},
        "failure_modes": ["backend_missing", "insufficient_replicates", "method_dependency_error"],
    },
    "deseq2_tissue_vs_rest": {
        "name": "deseq2_tissue_vs_rest",
        "aliases": ["DESeq2 target vs rest", "el vs rest", "组织差异表达"],
        "biological_intent": "Run differential expression for one target tissue/group against all remaining samples.",
        "description": "Fallback R/Bioconductor-oriented deterministic DESeq2 path when OmicVerse backend is unavailable or not desired.",
        "tool": "bio_deseq2_tissue_vs_rest",
        "parameters": {"count_matrix_path": "string", "target_tissue": "string", "padj_threshold": "number=0.05"},
        "requires": {
            "state": [
                "count_matrix_inspected",
                "sample_groups",
                "target_has_replicates",
                "rest_has_replicates",
                "deseq2_environment_checked",
            ]
        },
        "produces": {
            "files": [
                "deseq2_metadata_csv",
                "deseq2_r_script",
                "deseq2_full_results_csv",
                "deseq2_significant_csv",
                "deseq2_summary_json",
            ],
            "state": ["differential_expression_completed"],
        },
        "side_effects": ["writes analysis artifacts", "runs Rscript when environment passes"],
        "return_structure": {"status": "completed|failed|script_written", "summary": "dict"},
        "failure_modes": ["insufficient_replicates", "deseq2_runtime_error", "output_write_error"],
    },
    "lightweight_group_compare": {
        "name": "lightweight_group_compare",
        "aliases": ["welch compare", "descriptive comparison", "轻量比较"],
        "biological_intent": "Provide descriptive two-group comparison when DESeq2 is unavailable.",
        "description": "Run observed-count comparison without claiming publication-grade differential expression.",
        "tool": "bio_rnaseq_compare",
        "parameters": {"counts_path": "string", "metadata_path": "string", "method": "welch|deseq2"},
        "requires": {"files": ["count_matrix", "metadata"], "state": ["sample_groups"]},
        "produces": {"files": ["comparison_json"], "state": ["descriptive_comparison_completed"]},
        "side_effects": ["writes optional comparison output"],
        "return_structure": {"method_used": "string", "results": "list"},
        "failure_modes": ["missing_metadata", "invalid_group_column"],
    },
    "emit_transcriptome_report": {
        "name": "emit_transcriptome_report",
        "aliases": ["RNA-seq report", "转录组报告", "analysis summary"],
        "biological_intent": "Summarize observed RNA-seq analysis outputs without inventing biological interpretation.",
        "description": "Write a human-readable report from observed result JSON.",
        "tool": "bio_report",
        "parameters": {"result_json": "string", "output_path": "string"},
        "requires": {"state": ["differential_expression_completed"]},
        "produces": {"files": ["report_md"]},
        "side_effects": ["writes reports/*.md"],
        "return_structure": {"output_path": "string"},
        "failure_modes": ["missing_results", "report_path_policy_block"],
    },
}


def list_capabilities() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in CAPABILITIES.values()]


def get_capability(name: str) -> dict[str, Any]:
    key = str(name or "").strip()
    if key not in CAPABILITIES:
        raise KeyError(f"unknown transcriptome capability: {name}")
    return deepcopy(CAPABILITIES[key])
