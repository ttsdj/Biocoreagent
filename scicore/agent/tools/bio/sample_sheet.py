"""Sample metadata inspection and validation."""

from __future__ import annotations

import json
from collections import Counter

from ..base import Tool
from .tables import file_error, read_tsv


class BioSampleSheetInspectTool(Tool):
    name = "bio_sample_sheet_inspect"
    description = (
        "Inspect a TSV sample sheet. Requires a sample column and optionally checks "
        "a group/condition column. Does not infer missing metadata."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to sample metadata TSV."},
            "sample_column": {"type": "string", "description": "Sample id column. Default sample."},
            "group_column": {"type": "string", "description": "Group column. Default group."},
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, sample_column: str = "sample", group_column: str = "group") -> str:
        err = file_error(file_path)
        if err:
            return err
        try:
            rows = read_tsv(file_path)
            columns = list(rows[0].keys()) if rows else []
            missing = []
            if sample_column not in columns:
                missing.append(sample_column)
            if group_column and group_column not in columns:
                missing.append(group_column)

            samples = [row.get(sample_column, "").strip() for row in rows] if sample_column in columns else []
            duplicate_samples = sorted(sample for sample, count in Counter(samples).items() if sample and count > 1)
            blank_samples = sum(1 for sample in samples if not sample)
            group_counts = Counter(row.get(group_column, "").strip() for row in rows) if group_column in columns else Counter()
            if "" in group_counts:
                group_counts["(blank)"] = group_counts.pop("")

            result = {
                "file_path": file_path,
                "row_count": len(rows),
                "columns": columns,
                "sample_column": sample_column,
                "group_column": group_column,
                "missing_required_columns": missing,
                "duplicate_samples": duplicate_samples,
                "blank_sample_count": blank_samples,
                "group_counts": dict(group_counts),
                "can_use_for_group_comparison": not missing and not duplicate_samples and blank_samples == 0 and len([g for g in group_counts if g != "(blank)"]) >= 2,
                "notes": ["Observed metadata only; no sample groups were inferred."],
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {e}"
