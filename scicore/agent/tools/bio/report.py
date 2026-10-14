"""Markdown report generation for observed bioinformatics results."""

from __future__ import annotations

import json
from pathlib import Path

from ..base import Tool
from ...policy import PolicyDecision, evaluate_file_write


class BioReportTool(Tool):
    name = "bio_report"
    description = (
        "Create a Markdown report from observed SciCoreAgent result JSON. "
        "Writes only to reports/ by default and does not invent conclusions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "result_json": {"type": "string", "description": "Observed result JSON string from another tool."},
            "output_path": {"type": "string", "description": "Markdown output path. Must be under reports/."},
            "title": {"type": "string", "description": "Report title."},
        },
        "required": ["result_json", "output_path"],
    }

    def execute(self, result_json: str, output_path: str, title: str = "SciCoreAgent Report") -> str:
        try:
            output = Path(output_path).expanduser().resolve()
            reports_root = Path("reports").resolve()
            try:
                output.relative_to(reports_root)
            except ValueError:
                return "Blocked by policy: bio_report writes must stay under reports/"
            decision, reason = evaluate_file_write(str(output))
            if decision == PolicyDecision.BLOCK:
                return f"Blocked by policy: {reason}"

            data = json.loads(result_json)
            markdown = _render_report(title, data)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(markdown, encoding="utf-8")
            return json.dumps({"output_path": str(output), "bytes_written": len(markdown.encode("utf-8"))}, indent=2)
        except json.JSONDecodeError as e:
            return f"Error: result_json is not valid JSON: {e}"
        except Exception as e:
            return f"Error: {e}"


def _render_report(title: str, data: dict) -> str:
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "This report summarizes observed tool output only. It does not add clinical interpretation or unobserved biological conclusions.",
        "",
        "## Method Note",
        "",
        str(data.get("method_note", "No method note was provided.")),
        "",
        "## Observed Inputs",
        "",
    ]
    for key in ["case_group", "control_group", "case_samples", "control_samples"]:
        if key in data:
            lines.append(f"- **{key}**: {data[key]}")
    if "top_results" in data:
        lines.extend(["", "## Top Observed Results", "", "| Gene | Case mean | Control mean | log2FC | Welch t |", "|---|---:|---:|---:|---:|"])
        for row in data["top_results"]:
            lines.append(
                "| {gene} | {case_mean} | {control_mean} | {log2_fold_change} | {welch_t_statistic} |".format(
                    **row
                )
            )
    lines.extend(["", "## Limitations", "", "- No adjusted p-values are reported unless present in the observed input.", "- Treat this as a lightweight QA/demo report, not publication-grade differential expression analysis.", ""])
    return "\n".join(lines)
