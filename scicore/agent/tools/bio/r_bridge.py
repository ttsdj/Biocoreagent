"""Safe R bridge — execute small R scripts for bioinformatics analysis.

Runs user-supplied R code in a sandboxed Rscript subprocess with:
- 120s timeout
- 4000-character output cap
- Dangerous-function blocklist (system, file.remove, download.file, etc.)
- No network access, no file deletion
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..base import Tool

# -- Security -----------------------------------------------------------------

# Functions that are completely blocked
_BLOCKED_FUNCTIONS = [
    "system",
    "system2",
    "shell",
    "shell.exec",
    "file.remove",
    "file.delete",
    "unlink",
    "download.file",
    "url",
    "curl",
    "source",
    "install.packages",
    "update.packages",
    "remove.packages",
    "setwd",
    "quit",
    "q",
    ".Internal",
    ".Primitive",
    ".Call",
    ".C",
    ".Fortran",
    ".External",
    ".Call",
    ".External2",
    ".External.graphics",
    "save.image",
    "save",
]

# Packages that are safe to load
_ALLOWED_PACKAGES = [
    # Differential expression & normalization
    "DESeq2",
    "edgeR",
    "limma",
    "vsn",
    "sva",
    "RUVSeq",
    "zinbwave",
    # Single-cell
    "Seurat",
    "SeuratObject",
    "SingleCellExperiment",
    "scran",
    "scater",
    "scuttle",
    "scDblFinder",
    # Bioconductor core
    "BiocGenerics",
    "GenomicRanges",
    "SummarizedExperiment",
    "Biobase",
    "S4Vectors",
    "IRanges",
    "GenomeInfoDb",
    "GenomicFeatures",
    "Rsamtools",
    "rtracklayer",
    "Biostrings",
    "BSgenome",
    # Annotation
    "AnnotationDbi",
    "AnnotationFilter",
    "AnnotationHub",
    "org.Hs.eg.db",
    "org.Mm.eg.db",
    "org.Dr.eg.db",
    "org.Rn.eg.db",
    "org.Sc.sgd.db",
    "org.Dm.eg.db",
    "org.Ce.eg.db",
    "org.At.tair.db",
    "GO.db",
    "KEGGREST",
    "KEGG.db",
    "reactome.db",
    # Enrichment / pathway
    "clusterProfiler",
    "enrichplot",
    "DOSE",
    "fgsea",
    "msigdbr",
    "pathview",
    "GSEABase",
    "GSVA",
    # Visualization
    "ggplot2",
    "dplyr",
    "tidyr",
    "tibble",
    "readr",
    "stringr",
    "purrr",
    "forcats",
    "pheatmap",
    "RColorBrewer",
    "scales",
    "ggrepel",
    "ggpubr",
    "cowplot",
    "patchwork",
    "ComplexHeatmap",
    "circlize",
    "ggtree",
    "ape",
    "phytools",
    # Data
    "jsonlite",
    "data.table",
    "Matrix",
    "readxl",
    "writexl",
    # Stats / ML
    "glmnet",
    "survival",
    "survminer",
    "lme4",
    "lmerTest",
    "MASS",
    "car",
    "emmeans",
    "multcomp",
    "coin",
    "exact2x2",
    # Genomics
    "VariantAnnotation",
    "snpStats",
    "gwasurvivr",
    # Base R
    "stats",
    "graphics",
    "grDevices",
    "utils",
    "methods",
    "base",
]

# Pre-approved libraries auto-loaded
_AUTO_LIBRARIES = ["jsonlite"]

# Maximum R code length
_MAX_R_CODE_LENGTH = 4000

# Output cap
_MAX_OUTPUT_LENGTH = 4000


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


def _validate_r_code(code: str) -> str | None:
    """Check R code for dangerous function calls. Returns error message or None."""
    # Remove R comments for checking
    lines = []
    for line in code.splitlines():
        # Remove end-of-line comments (# ...)
        comment_pos = line.find("#")
        if comment_pos >= 0:
            # Check if # is inside a string (basic heuristic)
            pre = line[:comment_pos]
            if pre.count('"') % 2 == 0 and pre.count("'") % 2 == 0:
                line = pre
        lines.append(line)
    clean = "\n".join(lines)

    # Check for blocked function calls: func_name(
    for func in _BLOCKED_FUNCTIONS:
        pattern = rf"\b{re.escape(func)}\s*\("
        if re.search(pattern, clean):
            return f"Blocked function detected: {func}() is not allowed for security."

    # Check for `::` access to blocked namespaces
    if re.search(r":::\s*", clean):
        return "Triple-colon (:::) access to internal functions is not allowed."

    # Check for file write operations
    write_patterns = [
        r"\bwrite\.(csv|table|lines|bin)\s*\(",
        r"\bwriteLines\s*\(",
        r"\bsaveRDS\s*\(",
        r"\bggsave\s*\(",
        r"\bpdf\s*\(",
        r"\bpng\s*\(",
        r"\bjpeg\s*\(",
        r"\bsvg\s*\(",
        r"\btiff\s*\(",
    ]
    for pat in write_patterns:
        if re.search(pat, clean):
            return f"File write operation detected: {pat}. File creation is not allowed."

    # Check library() calls for allowed packages
    lib_matches = re.findall(r"library\s*\(\s*['\"]?(\w+)['\"]?\s*\)", clean)
    for pkg in lib_matches:
        if pkg not in _ALLOWED_PACKAGES:
            return (
                f"Package '{pkg}' is not in the allowed list. "
                f"Allowed packages: {', '.join(sorted(_ALLOWED_PACKAGES[:10]))}..."
            )

    # Also check require() calls
    req_matches = re.findall(r"require\s*\(\s*['\"]?(\w+)['\"]?\s*\)", clean)
    for pkg in req_matches:
        if pkg not in _ALLOWED_PACKAGES:
            return (
                f"Package '{pkg}' is not in the allowed list."
            )

    return None


def _wrap_r_code(code: str) -> str:
    """Wrap user R code with preamble and safe execution environment."""
    auto_loads = "\n".join(f"suppressMessages(library({lib}))" for lib in _AUTO_LIBRARIES)
    return f"""
# Auto-loaded helpers
options(warn = 1)
{auto_loads}

# Redirect output capture with detailed error feedback
output <- tryCatch({{
{code}
}}, error = function(e) {{
    cat("__R_BRIDGE_ERROR__\\n")
    cat("ERROR: ", conditionMessage(e), "\\n\\n")
    cat("Traceback:\\n")
    traceback(3)
    cat("\\n")
    cat("R version: ", R.version.string, "\\n")
    cat("Installed pkgs matching 'org.': ",
        paste(grep("org.", rownames(installed.packages()), value=TRUE), collapse=", "),
        "\\n")
}})

cat("__R_BRIDGE_DONE__\\n")
"""


class BioRBridgeTool(Tool):
    """Execute R scripts for bioinformatics analysis in a sandboxed environment.

    Restricted: max 4000 characters of R code, no file deletion, no network,
    only pre-approved packages (DESeq2, edgeR, Seurat, Bioconductor).
    Use absolute file paths — setwd() is blocked for security.

    Output is capped at 4000 characters.
    """

    name = "bio_r_bridge"
    description = (
        "Execute a sandboxed R script for bioinformatics analysis. "
        "Maximum 4000 characters of R code. Only pre-approved packages allowed "
        "(DESeq2, edgeR, Seurat, limma, ggplot2, dplyr, jsonlite, etc.). "
        "IMPORTANT: Use absolute file paths (e.g., read.csv(\"D:/data/file.csv\")) "
        "— do NOT use setwd() as it is blocked. "
        "No file creation, no network access, no system commands, no file deletion. "
        "Runs with a 120-second timeout. Output capped at 4000 characters. "
        "Use this for analyses beyond Python's capabilities: DESeq2, edgeR, "
        "Seurat operations, specialized Bioconductor functions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "r_code": {
                "type": "string",
                "description": (
                    "R code to execute (max 4000 chars). Use ABSOLUTE paths for files "
                    "(setwd() is blocked). The code runs in a tryCatch block. "
                    "Use cat() or print() to output results. "
                    "jsonlite::toJSON() is auto-loaded for JSON output. "
                    "Pre-approved packages: DESeq2, edgeR, Seurat, limma, ggplot2, "
                    "dplyr, jsonlite, Matrix, data.table."
                ),
            },
            "input_file": {
                "type": "string",
                "description": (
                    "Optional path to an input file (e.g., .rds, .csv) that the R code "
                    "can read. The file path is validated but not passed automatically — "
                    "reference it by the quoted path string in your R code."
                ),
            },
        },
        "required": ["r_code"],
    }

    def execute(self, r_code: str, input_file: str | None = None, timeout: int = 120) -> str:
        rscript = _find_rscript()
        if rscript is None:
            return json.dumps(
                {"error": "Rscript not found.", "detail": "R is required. Please install R."}
            )

        # Validate input file if provided
        if input_file:
            p = Path(input_file).expanduser().resolve()
            if not p.exists():
                return json.dumps({"error": f"Input file not found: {input_file}"})
            if not p.is_file():
                return json.dumps({"error": f"Input file is not a regular file: {input_file}"})

        # Trim and validate code
        r_code = r_code.strip()
        if len(r_code) > _MAX_R_CODE_LENGTH:
            return json.dumps(
                {
                    "error": f"R code exceeds maximum length of {_MAX_R_CODE_LENGTH} characters.",
                    "received_length": len(r_code),
                }
            )

        if not r_code:
            return json.dumps({"error": "R code is empty."})

        # Security validation
        error = _validate_r_code(r_code)
        if error:
            return json.dumps({"error": error})

        # Wrap with safety preamble
        wrapped = _wrap_r_code(r_code)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(wrapped)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [rscript, "--no-save", "--no-restore", "--no-environ", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"R script timed out after {timeout}s"})
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Process output
        stdout = result.stdout
        stderr = result.stderr.strip()

        # Check for R bridge error marker
        if "__R_BRIDGE_ERROR__" in stdout:
            error_parts = stdout.split("__R_BRIDGE_ERROR__", 1)
            error_msg = error_parts[1].replace("__R_BRIDGE_DONE__", "").strip()
            return json.dumps(
                {
                    "error": "R code execution error.",
                    "r_error": error_msg[:3000],
                }
            )

        if result.returncode != 0:
            # R process itself crashed
            error_lines = [l for l in stderr.splitlines() if "Error" in l or "error" in l]
            error_msg = error_lines[-1] if error_lines else stderr[-500:]
            return json.dumps(
                {
                    "error": "R execution failed.",
                    "r_exit_code": result.returncode,
                    "r_error": error_msg,
                }
            )

        # Clean up the output
        clean_output = stdout.replace("__R_BRIDGE_DONE__", "").strip()

        if len(clean_output) > _MAX_OUTPUT_LENGTH:
            clean_output = clean_output[:_MAX_OUTPUT_LENGTH] + "\n... [output truncated]"

        # Try to parse as JSON if it looks like JSON
        parsed_json = None
        if clean_output.strip().startswith("{") or clean_output.strip().startswith("["):
            try:
                parsed_json = json.loads(clean_output.strip())
            except json.JSONDecodeError:
                pass

        output = {
            "success": True,
            "output": clean_output if parsed_json is None else None,
            "json_output": parsed_json,
            "output_length": len(clean_output),
            "truncated": len(clean_output) >= _MAX_OUTPUT_LENGTH,
        }

        if stderr:
            output["r_warnings"] = stderr[:500]

        # Remove None fields for cleaner output
        output = {k: v for k, v in output.items() if v is not None}

        return json.dumps(output, indent=2)
