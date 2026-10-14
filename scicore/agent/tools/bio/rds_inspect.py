"""RDS file inspection via Rscript — extracts structure and summary statistics.

Reads .rds files (R serialized objects) without full deserialization into Python.
Works with DESeq2 results, Seurat objects, data.frame, list, matrix, and other
common bioinformatics R objects.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..base import Tool


# -- R snippet library ---------------------------------------------------------

def _r_inspect_script(file_path: str) -> str:
    """Generate an R script that loads an RDS and prints its structure."""
    # Escape backslashes for R string literals on Windows
    escaped = file_path.replace("\\", "\\\\")
    return rf"""
suppressMessages({{
    obj <- tryCatch(readRDS("{escaped}"), error = function(e) stop("Failed to read RDS: ", e$message))

    cat("__CLASS__\n")
    cat(class(obj), sep = ", ")
    cat("\n")

    cat("__MODE__\n")
    cat(typeof(obj))
    cat("\n")

    # Object length / dimensions
    cat("__LENGTH__\n")
    if (isS4(obj)) {{
        cat("S4_object")
    }} else if (is.list(obj) && !is.data.frame(obj)) {{
        cat(length(obj))
    }} else {{
        cat(length(obj))
    }}
    cat("\n")

    cat("__DIMENSIONS__\n")
    if (!is.null(dim(obj))) {{
        cat(paste(dim(obj), collapse = " x "))
    }} else if (is.vector(obj) || is.factor(obj)) {{
        cat(length(obj))
    }} else {{
        cat("scalar_or_null")
    }}
    cat("\n")

    cat("__NAMES__\n")
    nms <- names(obj)
    if (!is.null(nms)) {{
        cat(paste(head(nms, 200), collapse = "\t"))
    }} else {{
        cat("NULL")
    }}
    cat("\n")

    cat("__COLNAMES__\n")
    if (!is.null(colnames(obj))) {{
        cat(paste(head(colnames(obj), 200), collapse = "\t"))
    }} else {{
        cat("NULL")
    }}
    cat("\n")

    cat("__ROWNAMES_COUNT__\n")
    if (!is.null(rownames(obj))) {{
        cat(nrow(obj) %||% length(rownames(obj)) %||% 0)
    }} else {{
        cat(0)
    }}
    cat("\n")

    # Summary statistics for numeric columns (DESeq2 results, matrices)
    cat("__NUMERIC_SUMMARY__\n")
    numeric_summary <- list()
    if (is.data.frame(obj) || is.matrix(obj)) {{
        df <- as.data.frame(obj)
        for (cn in colnames(df)) {{
            if (is.numeric(df[[cn]])) {{
                vals <- df[[cn]][is.finite(df[[cn]])]
                if (length(vals) > 0) {{
                    numeric_summary[[cn]] <- c(
                        min    = signif(min(vals), 6),
                        q25    = signif(quantile(vals, 0.25, na.rm = TRUE), 6),
                        median = signif(median(vals), 6),
                        q75    = signif(quantile(vals, 0.75, na.rm = TRUE), 6),
                        max    = signif(max(vals), 6),
                        mean   = signif(mean(vals), 6),
                        nas    = sum(is.na(df[[cn]]))
                    )
                }}
            }}
        }}
    }}
    cat(gsub("\n", " ", jsonlite::toJSON(numeric_summary, auto_unbox = TRUE)))
    cat("\n")

    # Seurat-specific metadata
    cat("__SEURAT_INFO__\n")
    if (inherits(obj, "Seurat")) {{
        cat(paste("Seurat version:", as.character(tryCatch(obj@version, error = function(e) "unknown"))))
        cat("\n")
        cat(paste("Assay(s):", paste(names(obj@assays), collapse = ", ")))
        cat("\n")
        cat(paste("Active assay:", obj@active.assay))
        cat("\n")
        cat(paste("Dimensional reductions:", paste(names(obj@reductions), collapse = ", ")))
        cat("\n")
        cat(paste("Number of cells:", ncol(obj)))
        cat("\n")
        cat(paste("Number of features:", nrow(obj)))
        cat("\n")
    }} else {{
        cat("not_a_Seurat_object")
        cat("\n")
    }}

    cat("__SLOT_NAMES__\n")
    if (isS4(obj)) {{
        cat(paste(slotNames(obj), collapse = "\t"))
    }} else {{
        cat("not_S4")
    }}
    cat("\n")

    cat("__DONE__\n")
}})
"""
    return script


def _parse_r_inspect_output(stdout: str) -> dict:
    """Parse the sectioned output from the R inspection script."""
    result = {
        "class": [],
        "mode": "",
        "length": "",
        "dimensions": "",
        "names": [],
        "colnames": [],
        "rownames_count": 0,
        "numeric_summary": {},
        "seurat_info": None,
        "slot_names": [],
    }

    current_section = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        if line == "__CLASS__":
            current_section = "class"
            continue
        elif line == "__MODE__":
            current_section = "mode"
            continue
        elif line == "__LENGTH__":
            current_section = "length"
            continue
        elif line == "__DIMENSIONS__":
            current_section = "dimensions"
            continue
        elif line == "__NAMES__":
            current_section = "names"
            continue
        elif line == "__COLNAMES__":
            current_section = "colnames"
            continue
        elif line == "__ROWNAMES_COUNT__":
            current_section = "rownames_count"
            continue
        elif line == "__NUMERIC_SUMMARY__":
            current_section = "numeric_summary"
            continue
        elif line == "__SEURAT_INFO__":
            current_section = "seurat_info"
            result["seurat_info"] = {}
            continue
        elif line == "__SLOT_NAMES__":
            current_section = "slot_names"
            continue
        elif line == "__DONE__":
            break

        if current_section == "class":
            result["class"] = [c.strip() for c in line.split(",") if c.strip()]
        elif current_section == "mode":
            result["mode"] = line
        elif current_section == "length":
            result["length"] = line
        elif current_section == "dimensions":
            result["dimensions"] = line
        elif current_section == "names":
            result["names"] = [n.strip() for n in line.split("\t") if n.strip()] if line != "NULL" else []
        elif current_section == "colnames":
            result["colnames"] = [c.strip() for c in line.split("\t") if c.strip()] if line != "NULL" else []
        elif current_section == "rownames_count":
            try:
                result["rownames_count"] = int(line)
            except ValueError:
                result["rownames_count"] = line
        elif current_section == "numeric_summary":
            try:
                result["numeric_summary"] = json.loads(line)
            except json.JSONDecodeError:
                result["numeric_summary"] = {"parse_error": "could not decode numeric summary"}
        elif current_section == "seurat_info":
            if ":" in line and "not_a_Seurat_object" not in line:
                key, _, val = line.partition(": ")
                result["seurat_info"][key.strip()] = val.strip()
        elif current_section == "slot_names":
            if line != "not_S4":
                result["slot_names"] = [s.strip() for s in line.split("\t") if s.strip()]

    return result


def _find_rscript() -> str | None:
    """Locate the Rscript executable."""
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    # Windows common paths
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


# -- R helper format string used by both rds_inspect and r_bridge -------------

_R_HELPER_DECL = """
`%||%` <- function(x, y) if (is.null(x)) y else x
"""

# ---------------------------------------------------------------------------


class BioRDSInspectTool(Tool):
    """Read R .rds files and extract summary statistics without full deserialization.

    Uses Rscript to run a small R snippet that loads the RDS and prints structure.
    Works with: DESeq2 results, Seurat objects, data.frame, list, matrix.

    Never outputs full data — only metadata and summary statistics.
    """

    name = "bio_rds_inspect"
    description = (
        "Inspect an R .rds file (R serialized object) and extract its structure: "
        "object class, dimensions, column names, numeric summary statistics, "
        "and Seurat-specific metadata if applicable. "
        "Supports DESeq2 results, Seurat objects, data.frame, list, and matrix. "
        "Never outputs raw biological data — only structural metadata and summary stats."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the .rds file to inspect.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, timeout: int = 120) -> str:
        rscript = _find_rscript()
        if rscript is None:
            return json.dumps(
                {"error": "Rscript not found.", "detail": "R is required to read .rds files. Please install R."}
            )

        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return json.dumps({"error": f"File not found: {file_path}"})
        if not p.is_file():
            return json.dumps({"error": f"Not a file: {file_path}"})
        if p.suffix.lower() not in (".rds", ".RDS", ".rda", ".RDA", ".rdata", ".RDATA"):
            return json.dumps(
                {
                    "warning": f"File extension '{p.suffix}' is not a standard RDS extension. "
                    "Attempting to read anyway.",
                }
            )

        script = _R_HELPER_DECL + _r_inspect_script(str(p))

        # Write script to a temp file to avoid command-line escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(script)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [rscript, "--no-save", "--no-restore", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"RDS inspection timed out after {timeout}s"})
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Extract the most useful error line
            error_lines = [l for l in stderr.splitlines() if "Error" in l or "error" in l]
            error_msg = error_lines[-1] if error_lines else stderr[-500:]
            return json.dumps(
                {
                    "error": "R execution failed.",
                    "r_exit_code": result.returncode,
                    "r_error": error_msg,
                }
            )

        try:
            parsed = _parse_r_inspect_output(result.stdout)
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to parse R output: {e}",
                    "raw_output_preview": result.stdout[:1000],
                }
            )

        # Build a clean, structured output
        output = {
            "file_path": str(p),
            "object_class": parsed["class"],
            "type": parsed["mode"],
            "dimensions": parsed["dimensions"],
            "column_count": len(parsed["colnames"]),
            "columns": parsed["colnames"][:50],  # cap at 50 column names
            "columns_truncated": len(parsed["colnames"]) > 50,
            "rownames_count": parsed["rownames_count"],
            "numeric_summary": parsed["numeric_summary"],
        }

        if parsed["names"]:
            output["top_level_names"] = parsed["names"][:100]

        if parsed["seurat_info"]:
            output["seurat_info"] = parsed["seurat_info"]

        if parsed["slot_names"]:
            output["s4_slot_names"] = parsed["slot_names"]

        return json.dumps(output, indent=2)
