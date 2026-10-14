"""Lightweight RNA-seq count comparison with optional DESeq2 backend.

Provides two methods:
- "welch": Pure-Python Welch t-test (always available, no R required)
- "deseq2": Runs DESeq2 via Rscript for proper differential expression analysis

When method="deseq2" is requested but R/DESeq2 is unavailable, the tool
automatically falls back to the Welch method with a clear warning.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import mean, variance

from ..base import Tool
from .tables import file_error, read_count_matrix, read_tsv


def _find_rscript() -> str | None:
    """Locate the Rscript executable."""
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    for base in [r"C:\Program Files\R", r"C:\Program Files (x86)\R"]:
        if os.path.isdir(base):
            for entry in os.listdir(base):
                candidate = os.path.join(base, entry, "bin", "x64", "Rscript.exe")
                if os.path.isfile(candidate):
                    return candidate
                candidate = os.path.join(base, entry, "bin", "Rscript.exe")
                if os.path.isfile(candidate):
                    return candidate
    return None


def _check_deseq2_available(rscript: str) -> bool:
    """Check if DESeq2 package is installed in the R environment."""
    try:
        result = subprocess.run(
            [rscript, "--no-save", "--no-restore", "-e",
             "cat(suppressMessages(requireNamespace('DESeq2', quietly=TRUE)))"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() == "TRUE"
    except Exception:
        return False


def _run_deseq2(
    count_matrix_path: str,
    metadata_path: str,
    case_group: str,
    control_group: str,
    group_column: str,
    sample_column: str,
    top_n: int,
    rscript: str,
    timeout: int = 120,
) -> str:
    """Run DESeq2 via Rscript and return parsed results as JSON string."""
    escaped_matrix = count_matrix_path.replace("\\", "\\\\")
    escaped_metadata = metadata_path.replace("\\", "\\\\")

    r_script = f"""
suppressMessages(library(DESeq2))

counts <- as.matrix(read.table("{escaped_matrix}", header=TRUE, row.names=1, sep="\\t", check.names=FALSE))
metadata <- read.table("{escaped_metadata}", header=TRUE, sep="\\t", stringsAsFactors=FALSE, check.names=FALSE)

rownames(metadata) <- metadata[["{sample_column}"]]
metadata[["{group_column}"]] <- factor(metadata[["{group_column}"]], levels=c("{control_group}", "{case_group}"))

samples <- intersect(colnames(counts), rownames(metadata))
counts <- counts[, samples, drop=FALSE]
metadata <- metadata[samples, , drop=FALSE]

dds <- DESeqDataSetFromMatrix(countData=counts, colData=metadata, design=as.formula("~ {group_column}"))
dds <- DESeq(dds)
res <- results(dds, contrast=c("{group_column}", "{case_group}", "{control_group}"))
res <- res[order(res$padj, na.last=TRUE), , drop=FALSE]

top <- head(res, {top_n})
out <- data.frame(
    gene = rownames(top),
    baseMean = signif(top$baseMean, 6),
    log2FoldChange = signif(top$log2FoldChange, 6),
    lfcSE = signif(top$lfcSE, 6),
    stat = signif(top$stat, 6),
    pvalue = signif(top$pvalue, 6),
    padj = signif(top$padj, 6),
    stringsAsFactors = FALSE
)
out <- out[order(out$padj), ]
cat(jsonlite::toJSON(list(
    method = "DESeq2",
    case_group = "{case_group}",
    control_group = "{control_group}",
    case_samples = as.character(metadata[samples, "{group_column}"]),
    control_samples = as.character(metadata[samples, "{group_column}"]),
    total_genes = nrow(res),
    significant_genes_005 = sum(res$padj < 0.05, na.rm=TRUE),
    significant_genes_001 = sum(res$padj < 0.01, na.rm=TRUE),
    top_results = out
), auto_unbox=TRUE))
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(r_script)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [rscript, "--no-save", "--no-restore", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"DESeq2 analysis timed out after {timeout}s"})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_lines = [l for l in stderr.splitlines() if "Error" in l or "error" in l]
        error_msg = error_lines[-1] if error_lines else stderr[-500:]
        return json.dumps(
            {
                "error": "DESeq2 execution failed.",
                "r_exit_code": result.returncode,
                "r_error": error_msg,
            }
        )

    # Find JSON in output (skip R startup messages)
    stdout = result.stdout
    json_start = stdout.find("{")
    if json_start == -1:
        return json.dumps({"error": "DESeq2 produced no JSON output", "raw": stdout[:500]})

    try:
        return stdout[json_start:].strip()
    except Exception:
        return json.dumps({"error": "Failed to extract DESeq2 output"})


class BioRNASeqCompareTool(Tool):
    name = "bio_rnaseq_compare"
    description = (
        "Run a two-group comparison on an observed count matrix and sample sheet. "
        "Supports two methods: 'welch' (pure Python Welch t-test, always available) "
        "and 'deseq2' (DESeq2 via R, requires R and DESeq2 package installed). "
        "When deseq2 is requested but unavailable, falls back to welch with a warning. "
        "Outputs descriptive statistics, log2 fold change, and for DESeq2: adjusted p-values."
    )
    parameters = {
        "type": "object",
        "properties": {
            "count_matrix_path": {"type": "string", "description": "TSV count matrix; first column is gene id."},
            "metadata_path": {"type": "string", "description": "TSV sample metadata."},
            "group_column": {"type": "string", "description": "Metadata group column. Default group."},
            "sample_column": {"type": "string", "description": "Metadata sample column. Default sample."},
            "case_group": {"type": "string", "description": "Case/treatment group label."},
            "control_group": {"type": "string", "description": "Control group label."},
            "top_n": {"type": "integer", "description": "Number of top rows to return. Default 20."},
            "method": {
                "type": "string",
                "description": "Analysis method: 'welch' (default) or 'deseq2'. Falls back to welch if deseq2 unavailable.",
                "enum": ["welch", "deseq2"],
            },
        },
        "required": ["count_matrix_path", "metadata_path", "case_group", "control_group"],
    }

    def execute(
        self,
        count_matrix_path: str,
        metadata_path: str,
        case_group: str,
        control_group: str,
        group_column: str = "group",
        sample_column: str = "sample",
        top_n: int = 20,
        method: str = "welch",
    ) -> str:
        for path in [count_matrix_path, metadata_path]:
            err = file_error(path)
            if err:
                return err
        if top_n <= 0:
            return "Error: top_n must be positive"
        if method not in ("welch", "deseq2"):
            return f"Error: unknown method '{method}'. Use 'welch' or 'deseq2'."

        # -- DESeq2 path ----------------------------------------------------------
        if method == "deseq2":
            rscript = _find_rscript()
            if rscript is None:
                return json.dumps(
                    {
                        "can_compare": True,
                        "method_used": "welch",
                        "method_requested": "deseq2",
                        "fallback_reason": "Rscript not found. R is required for DESeq2.",
                        "warning": "Falling back to Welch t-test. Install R and DESeq2 for proper differential expression analysis.",
                        "results": self._run_welch(
                            count_matrix_path, metadata_path, case_group, control_group,
                            group_column, sample_column, top_n,
                        ),
                    },
                    indent=2,
                )
            if not _check_deseq2_available(rscript):
                return json.dumps(
                    {
                        "can_compare": True,
                        "method_used": "welch",
                        "method_requested": "deseq2",
                        "fallback_reason": "DESeq2 R package is not installed.",
                        "warning": "Falling back to Welch t-test. Install DESeq2 with: BiocManager::install('DESeq2')",
                        "results": self._run_welch(
                            count_matrix_path, metadata_path, case_group, control_group,
                            group_column, sample_column, top_n,
                        ),
                    },
                    indent=2,
                )
            try:
                deseq2_output = _run_deseq2(
                    count_matrix_path, metadata_path, case_group, control_group,
                    group_column, sample_column, top_n, rscript,
                )
                return deseq2_output
            except Exception as e:
                return json.dumps(
                    {
                        "can_compare": True,
                        "method_used": "welch",
                        "method_requested": "deseq2",
                        "fallback_reason": f"DESeq2 execution failed: {e}",
                        "warning": "Falling back to Welch t-test.",
                        "results": self._run_welch(
                            count_matrix_path, metadata_path, case_group, control_group,
                            group_column, sample_column, top_n,
                        ),
                    },
                    indent=2,
                )

        # -- Welch path -----------------------------------------------------------
        return json.dumps(
            self._run_welch(
                count_matrix_path,
                metadata_path,
                case_group,
                control_group,
                group_column,
                sample_column,
                top_n,
                method_used="welch",
            ),
            indent=2,
        )

    def _run_welch(
        self,
        count_matrix_path: str,
        metadata_path: str,
        case_group: str,
        control_group: str,
        group_column: str = "group",
        sample_column: str = "sample",
        top_n: int = 20,
        method_used: str = "welch",
    ) -> dict:
        """Run the lightweight Welch t-test comparison (pure Python)."""
        try:
            samples, counts = read_count_matrix(count_matrix_path)
            metadata = read_tsv(metadata_path)
            sample_to_group = {}
            for row in metadata:
                sample = row.get(sample_column, "").strip()
                group = row.get(group_column, "").strip()
                if sample:
                    sample_to_group[sample] = group

            missing_metadata = [sample for sample in samples if sample not in sample_to_group]
            case_samples = [sample for sample in samples if sample_to_group.get(sample) == case_group]
            control_samples = [sample for sample in samples if sample_to_group.get(sample) == control_group]
            missing = []
            if missing_metadata:
                missing.append({"missing_metadata_for_samples": missing_metadata})
            if not case_samples:
                missing.append({"missing_case_group": case_group})
            if not control_samples:
                missing.append({"missing_control_group": control_group})
            if missing:
                return {
                    "can_compare": False,
                    "method_used": method_used,
                    "case_group": case_group,
                    "control_group": control_group,
                    "case_samples": case_samples,
                    "control_samples": control_samples,
                    "missing_information": missing,
                }

            results = []
            for row in counts:
                case_values = [row[sample] for sample in case_samples]
                control_values = [row[sample] for sample in control_samples]
                case_mean = mean(case_values)
                control_mean = mean(control_values)
                log2fc = math.log2((case_mean + 1.0) / (control_mean + 1.0))
                t_stat = _welch_t(case_values, control_values)
                results.append(
                    {
                        "gene": row["gene"],
                        "case_mean": round(case_mean, 4),
                        "control_mean": round(control_mean, 4),
                        "log2_fold_change": round(log2fc, 4),
                        "welch_t_statistic": round(t_stat, 4) if t_stat is not None else None,
                    }
                )
            results.sort(key=lambda item: abs(item["log2_fold_change"]), reverse=True)
            return {
                "can_compare": True,
                "method_used": method_used,
                "case_group": case_group,
                "control_group": control_group,
                "case_samples": case_samples,
                "control_samples": control_samples,
                "method_note": "Lightweight descriptive comparison only; not a DESeq2/edgeR replacement and no adjusted p-values are reported.",
                "top_results": results[:top_n],
            }
        except Exception as e:
            return {"error": str(e)}


def _welch_t(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va = variance(a)
    vb = variance(b)
    denom = math.sqrt(va / len(a) + vb / len(b))
    if denom == 0:
        return None
    return (mean(a) - mean(b)) / denom
