"""Count matrix inspection for RNA-seq style tables."""

from __future__ import annotations

import json

from ..base import Tool
from .tables import file_error, read_count_matrix


class BioCountMatrixInspectTool(Tool):
    name = "bio_count_matrix_inspect"
    description = "Inspect a TSV gene-by-sample count matrix without generating biological conclusions."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to count matrix TSV."},
            "max_genes": {"type": "integer", "description": "Maximum genes to scan. Default 10000."},
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, max_genes: int = 10000) -> str:
        err = file_error(file_path)
        if err:
            return err
        if max_genes <= 0:
            return "Error: max_genes must be positive"
        try:
            samples, rows = read_count_matrix(file_path)
            truncated = len(rows) > max_genes
            scanned = rows[:max_genes]
            total_counts = {sample: sum(row[sample] for row in scanned) for sample in samples}
            zero_count_genes = sum(1 for row in scanned if all(row[sample] == 0 for sample in samples))
            result = {
                "file_path": file_path,
                "gene_count_seen": len(scanned),
                "sample_count": len(samples),
                "samples": samples,
                "total_counts_seen": total_counts,
                "zero_count_genes_seen": zero_count_genes,
                "truncated": truncated,
                "notes": [
                    "Counts are inspected as provided; no normalization or differential-expression claim is made.",
                    "Use bio_rnaseq_compare for a lightweight descriptive comparison when metadata is available.",
                ],
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {e}"
