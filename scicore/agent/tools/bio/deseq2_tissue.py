"""Pre-validated DESeq2 tissue-vs-rest workflow."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

from scicore.agent.tools.base import Tool
from scicore.agent.tools.bio.r_bridge import _find_rscript


class BioDESeq2TissueVsRestTool(Tool):
    name = "bio_deseq2_tissue_vs_rest"
    description = (
        "Run a pre-validated DESeq2 comparison for count matrices whose sample columns "
        "encode tissue prefixes, e.g. el1/el2/el3 vs all other tissues. The tool infers "
        "sample metadata, writes metadata and an R script, runs Rscript when available, "
        "and saves full/significant result CSV files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "count_matrix_path": {
                "type": "string",
                "description": "TSV count matrix with gene IDs in the first column and sample count columns.",
            },
            "target_tissue": {
                "type": "string",
                "description": "Tissue prefix to compare against all other tissues, e.g. el.",
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory. Defaults to the count matrix directory.",
            },
            "padj_threshold": {"type": "number", "default": 0.05},
            "timeout": {"type": "integer", "default": 600},
        },
        "required": ["count_matrix_path", "target_tissue"],
    }

    def execute(
        self,
        count_matrix_path: str,
        target_tissue: str,
        output_dir: str = "",
        padj_threshold: float = 0.05,
        timeout: int = 600,
    ) -> str:
        matrix = Path(count_matrix_path).resolve()
        if not matrix.exists():
            return json.dumps({"error": f"count matrix not found: {matrix}"}, ensure_ascii=False, indent=2)
        target = str(target_tissue).strip().lower()
        if not target:
            return json.dumps({"error": "target_tissue is required"}, ensure_ascii=False, indent=2)

        samples = _read_sample_columns(matrix)
        tissues = [_sample_prefix(sample) for sample in samples]
        target_samples = [sample for sample, tissue in zip(samples, tissues) if tissue == target]
        other_samples = [sample for sample, tissue in zip(samples, tissues) if tissue != target]
        if len(target_samples) < 2:
            return json.dumps(
                {
                    "error": "DESeq2 requires biological replicates for the target tissue.",
                    "target_tissue": target,
                    "target_samples": target_samples,
                    "sample_columns": samples,
                },
                ensure_ascii=False,
                indent=2,
            )
        if len(other_samples) < 2:
            return json.dumps(
                {
                    "error": "DESeq2 requires at least two non-target samples for the rest group.",
                    "target_tissue": target,
                    "other_samples": other_samples,
                },
                ensure_ascii=False,
                indent=2,
            )

        out = Path(output_dir).resolve() if output_dir else matrix.parent
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"deseq2_{target}_vs_rest"
        metadata_path = out / f"{prefix}_metadata.csv"
        script_path = out / f"{prefix}.R"
        full_csv = out / f"{prefix}_full_results.csv"
        sig_csv = out / f"{prefix}_significant.csv"
        summary_json = out / f"{prefix}_summary.json"

        _write_metadata(metadata_path, samples, tissues, target)
        script = _render_r_script(
            matrix=matrix,
            metadata=metadata_path,
            target=target,
            full_csv=full_csv,
            sig_csv=sig_csv,
            summary_json=summary_json,
            padj_threshold=float(padj_threshold),
        )
        script_path.write_text(script, encoding="utf-8")

        rscript = _find_rscript()
        base_result = {
            "target_tissue": target,
            "target_samples": target_samples,
            "other_sample_count": len(other_samples),
            "metadata_path": str(metadata_path),
            "script_path": str(script_path),
            "full_results_path": str(full_csv),
            "significant_results_path": str(sig_csv),
            "summary_path": str(summary_json),
        }
        if rscript is None:
            return json.dumps({**base_result, "status": "script_written", "error": "Rscript not found"}, ensure_ascii=False, indent=2)

        proc = subprocess.run(
            [rscript, "--no-save", "--no-restore", "--no-environ", str(script_path)],
            cwd=str(out),
            capture_output=True,
            text=True,
            timeout=int(timeout or 600),
        )
        result = {
            **base_result,
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        }
        if summary_json.exists():
            try:
                result["summary"] = json.loads(summary_json.read_text(encoding="utf-8"))
            except Exception:
                result["summary_parse_error"] = True
        return json.dumps(result, ensure_ascii=False, indent=2)


def _read_sample_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if len(header) < 3:
        raise ValueError("count matrix must contain a gene column and at least two samples")
    return [col.strip() for col in header[1:] if col.strip()]


def _sample_prefix(sample: str) -> str:
    match = re.match(r"([A-Za-z]+)", sample.strip())
    return match.group(1).lower() if match else sample.lower()


def _write_metadata(path: Path, samples: list[str], tissues: list[str], target: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "tissue", "condition"])
        for sample, tissue in zip(samples, tissues):
            writer.writerow([sample, tissue, target if tissue == target else "rest"])


def _r_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _render_r_script(
    matrix: Path,
    metadata: Path,
    target: str,
    full_csv: Path,
    sig_csv: Path,
    summary_json: Path,
    padj_threshold: float,
) -> str:
    return f'''options(warn = 1)
suppressMessages(library(DESeq2))
suppressMessages(library(jsonlite))

count_data <- read.table("{_r_path(matrix)}", header=TRUE, row.names=1, sep="\\t", check.names=FALSE)
meta <- read.csv("{_r_path(metadata)}", row.names=1, check.names=FALSE)
count_data <- count_data[, rownames(meta), drop=FALSE]
meta$condition <- factor(meta$condition, levels=c("rest", "{target}"))

dds <- DESeqDataSetFromMatrix(countData=round(as.matrix(count_data)), colData=meta, design=~condition)
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep,]
dds <- DESeq(dds)
res <- results(dds, contrast=c("condition", "{target}", "rest"), alpha={padj_threshold})
res_df <- as.data.frame(res[order(res$padj),])
res_df$gene <- rownames(res_df)
res_df <- res_df[, c("gene", setdiff(colnames(res_df), "gene"))]
sig <- res_df[!is.na(res_df$padj) & res_df$padj < {padj_threshold},]

write.csv(res_df, "{_r_path(full_csv)}", row.names=FALSE, quote=FALSE)
write.csv(sig, "{_r_path(sig_csv)}", row.names=FALSE, quote=FALSE)

summary <- list(
  target_tissue="{target}",
  total_genes=nrow(res_df),
  tested_genes=nrow(res_df),
  significant_genes=nrow(sig),
  up_in_target=sum(sig$log2FoldChange > 0, na.rm=TRUE),
  down_in_target=sum(sig$log2FoldChange < 0, na.rm=TRUE),
  padj_threshold={padj_threshold},
  top_genes=head(res_df, 20)
)
write(jsonlite::toJSON(summary, auto_unbox=TRUE, pretty=TRUE, digits=6), "{_r_path(summary_json)}")
cat(jsonlite::toJSON(summary, auto_unbox=TRUE, pretty=TRUE, digits=6))
'''
