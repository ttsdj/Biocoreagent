"""Pre-validated bioinformatics analysis tools for BixBench.

OmicOS-inspired: agent selects tools and provides parameters,
NOT write R code from scratch. Each tool internally runs
validated R code and returns structured results.
"""

from __future__ import annotations

import json
import os
import tempfile
import subprocess
from pathlib import Path

from scicore.agent.tools.base import Tool

# Reuse the R bridge infrastructure
from scicore.agent.tools.bio.r_bridge import (
    _find_rscript,
    _validate_r_code,
    _MAX_R_CODE_LENGTH,
    _MAX_OUTPUT_LENGTH,
)


def _run_r_script(r_code: str, timeout: int = 120) -> dict:
    """Execute R code and return parsed result dict."""
    rscript = _find_rscript()
    if rscript is None:
        return {"error": "Rscript not found"}

    r_code = r_code.strip()
    if len(r_code) > _MAX_R_CODE_LENGTH:
        return {"error": f"Code too long: {len(r_code)} > {_MAX_R_CODE_LENGTH}"}

    err = _validate_r_code(r_code)
    if err:
        return {"error": err}

    # Wrap with safety
    wrapped = f"""
options(warn = 1)
suppressMessages(library(jsonlite))

output <- tryCatch({{
{r_code}
}}, error = function(e) {{
    cat(jsonlite::toJSON(list(error=TRUE, message=conditionMessage(e)), auto_unbox=TRUE))
}})
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(wrapped)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [rscript, "--no-save", "--no-restore", "--no-environ", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout.strip()

        # Try to extract JSON result from marker
        if "---RESULT---" in stdout:
            parts = stdout.split("---RESULT---", 1)
            diagnostic = parts[0].strip()
            json_str = parts[1].strip()
            try:
                parsed = json.loads(json_str)
                parsed["_diagnostic"] = diagnostic[-500:]
                return parsed
            except json.JSONDecodeError:
                return {"output": diagnostic[-1000:], "raw_json": json_str[:500]}

        if stdout:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                return {"output": stdout[:2000]}
        return {"error": "No output"}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Tool 1: Contingency Table Test ──────────────────────────────────────────

class BioContingencyTestTool(Tool):
    """Run chi-square / Fisher's exact test on contingency tables from CSV data.

    OmicOS-style: agent provides column names, tool runs validated R code.
    """

    name = "bio_contingency_test"
    description = (
        "Run a statistical test (chi-square or Fisher's exact) on categorical "
        "data from CSV files. Specify the file path, the grouping column (e.g., "
        "treatment vs placebo), the outcome column (e.g., severity), and "
        "optionally a filter column/value to subset the data. "
        "Returns: test statistic, p-value, contingency table. "
        "Use this for questions like 'is there a significant association between "
        "X and Y in group Z?'"
    )
    parameters = {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Absolute path to the CSV data file."
            },
            "group_col": {
                "type": "string",
                "description": "Column name for the grouping variable (e.g., treatment group)."
            },
            "outcome_col": {
                "type": "string",
                "description": "Column name for the outcome variable (e.g., severity score)."
            },
            "filter_col": {
                "type": "string",
                "description": "Optional: column name to filter on."
            },
            "filter_value": {
                "type": "string",
                "description": "Optional: value to match in filter_col."
            },
            "test_type": {
                "type": "string",
                "enum": ["chi_square", "fisher"],
                "description": "Statistical test to use. Default: chi_square with simulation."
            },
            "merge_csv_path": {
                "type": "string",
                "description": "Optional: second CSV to merge on USUBJID before testing."
            },
            "merge_filter_col": {
                "type": "string",
                "description": "Optional: filter the merge CSV on this column before merging."
            },
            "merge_filter_pattern": {
                "type": "string",
                "description": "Optional: grep pattern to filter merge CSV rows (e.g., 'COVID' for COVID-19 AEs)."
            },
        },
        "required": ["csv_path", "group_col", "outcome_col"],
    }

    def execute(self, csv_path: str, group_col: str, outcome_col: str,
                filter_col: str | None = None, filter_value: str | None = None,
                test_type: str = "chi_square",
                merge_csv_path: str | None = None,
                merge_filter_col: str | None = None,
                merge_filter_pattern: str | None = None) -> str:
        csv_path = csv_path.replace("\\", "/")

        # Build R code
        r_code = f'df <- read.csv("{csv_path}", stringsAsFactors=FALSE)\n'

        if merge_csv_path:
            mp = merge_csv_path.replace("\\", "/")
            r_code += f'df2 <- read.csv("{mp}", stringsAsFactors=FALSE)\n'
            if merge_filter_col and merge_filter_pattern:
                r_code += f'df2 <- df2[grep("{merge_filter_pattern}", df2${merge_filter_col}, ignore.case=TRUE), ]\n'
                r_code += f'cat("Filtered merge CSV to", nrow(df2), "rows matching", "{merge_filter_pattern}", "in", "{merge_filter_col}", "\\n")\n'
            r_code += f'df <- merge(df, df2, by="USUBJID", suffixes=c("", ".m"))\n'

        if filter_col and filter_value:
            r_code += f'df <- df[df${filter_col} == "{filter_value}", ]\n'
            r_code += f'cat("Filtered N:", nrow(df), "\\n")\n'

        r_code += f'''
cat("Group levels:", paste(unique(df${group_col}), collapse=", "), "\\n")
cat("Outcome levels:", paste(unique(df${outcome_col}), collapse=", "), "\\n")
tbl <- table(df${group_col}, df${outcome_col})
cat("Contingency table:\\n")
print(tbl)

if("{test_type}" == "chi_square") {{
    ct <- chisq.test(tbl, simulate.p.value=TRUE, B=10000)
}} else {{
    ct <- fisher.test(tbl, simulate.p.value=TRUE, B=10000)
}}

result <- list(
    test = "{test_type}",
    statistic = as.numeric(ct$statistic),
    p_value = as.numeric(ct$p.value),
    method = ct$method,
    table_dim = dim(tbl),
    n_total = sum(tbl)
)
cat("\\n---RESULT---\\n")
cat("P-VALUE: ", round(result$p_value, 6), " | TEST: {test_type} | N: ", result$n_total, " | GROUPS: ", result$table_dim[1], "x", result$table_dim[2], "\\n")
cat("\\nJSON:\\n")
cat(jsonlite::toJSON(result, auto_unbox=TRUE, pretty=TRUE))
'''

        res = _run_r_script(r_code, timeout=120)
        return json.dumps(res, ensure_ascii=False)


# ── Tool 2: Regression Analysis ─────────────────────────────────────────────

class BioRegressionTool(Tool):
    """Run logistic or linear regression on CSV data."""

    name = "bio_regression"
    description = (
        "Run logistic or ordinal regression on CSV data. "
        "Specify the formula (R-style: 'outcome ~ predictor1 + predictor2'), "
        "the CSV file path, and regression type. "
        "Returns: coefficients, odds ratios, p-values. "
        "Use for questions about odds ratios, risk factors, or associations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Absolute path to the CSV data file."
            },
            "formula": {
                "type": "string",
                "description": "R-style formula, e.g., 'AESEV ~ TRTGRP + Age'."
            },
            "regression_type": {
                "type": "string",
                "enum": ["logistic", "ordinal", "linear"],
                "description": "Type of regression. 'ordinal' uses polr() from MASS."
            },
            "filter_col": {
                "type": "string",
                "description": "Optional: column name to filter on."
            },
            "filter_value": {
                "type": "string",
                "description": "Optional: value to match in filter_col."
            },
        },
        "required": ["csv_path", "formula"],
    }

    def execute(self, csv_path: str, formula: str,
                regression_type: str = "logistic",
                filter_col: str | None = None,
                filter_value: str | None = None) -> str:
        csv_path = csv_path.replace("\\", "/")

        r_code = f'df <- read.csv("{csv_path}", stringsAsFactors=FALSE)\n'

        if filter_col and filter_value:
            r_code += f'df <- df[df${filter_col} == "{filter_value}", ]\n'
            r_code += f'cat("Filtered N:", nrow(df), "\\n")\n'

        if regression_type == "logistic":
            r_code += f'''
df$outcome_binary <- as.numeric(as.factor(df${formula.split("~")[0].strip()}))
m <- glm({formula}, data=df, family=binomial())
s <- summary(m)
coefs <- as.data.frame(s$coefficients)
coefs$odds_ratio <- exp(coefs[, 1])
coefs$ci_lower <- exp(coefs[, 1] - 1.96 * coefs[, 2])
coefs$ci_upper <- exp(coefs[, 1] + 1.96 * coefs[, 2])
result <- list(
    type = "logistic",
    formula = "{formula}",
    n = nrow(df),
    aic = s$aic,
    coefficients = coefs
)
'''
        elif regression_type == "ordinal":
            r_code += f'''
library(MASS)
df$outcome_factor <- as.factor(df${formula.split("~")[0].strip()})
m <- polr({formula}, data=df, Hess=TRUE)
s <- summary(m)
ctable <- coef(s)
result <- list(
    type = "ordinal",
    formula = "{formula}",
    n = nrow(df),
    coefficients = as.data.frame(ctable)
)
'''
        else:  # linear
            r_code += f'''
m <- lm({formula}, data=df)
s <- summary(m)
result <- list(
    type = "linear",
    formula = "{formula}",
    n = nrow(df),
    r_squared = s$r.squared,
    coefficients = as.data.frame(s$coefficients)
)
'''

        r_code += '''
cat("\\n---RESULT---\\n")
cat(jsonlite::toJSON(result, auto_unbox=TRUE, pretty=TRUE, digits=6))
'''

        res = _run_r_script(r_code, timeout=120)
        return json.dumps(res, ensure_ascii=False)


# ── Tool 3: Quick DESeq2 ────────────────────────────────────────────────────

class BioDESeq2QuickTool(Tool):
    """Run DESeq2 differential expression analysis with minimal configuration."""

    name = "bio_deseq2_quick"
    description = (
        "Run DESeq2 differential expression analysis. Provide paths to the "
        "count matrix (TSV/CSV, genes x samples) and metadata (CSV, samples x conditions). "
        "Specify the design column in metadata. "
        "Returns: number of significant DEGs, top genes with p-values. "
        "Use for RNA-seq differential expression questions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "counts_path": {
                "type": "string",
                "description": "Absolute path to count matrix file (genes in rows, samples in columns)."
            },
            "metadata_path": {
                "type": "string",
                "description": "Absolute path to sample metadata CSV."
            },
            "design_col": {
                "type": "string",
                "description": "Column name in metadata for the condition/group."
            },
            "padj_threshold": {
                "type": "number",
                "description": "Adjusted p-value threshold (default: 0.05)."
            },
            "reference_level": {
                "type": "string",
                "description": "Optional: reference condition level."
            },
        },
        "required": ["counts_path", "metadata_path", "design_col"],
    }

    def execute(self, counts_path: str, metadata_path: str, design_col: str,
                padj_threshold: float = 0.05,
                reference_level: str | None = None) -> str:
        counts_path = counts_path.replace("\\", "/")
        metadata_path = metadata_path.replace("\\", "/")

        r_code = f'''
library(DESeq2)

cts <- read.table("{counts_path}", header=TRUE, row.names=1, check.names=FALSE)
meta <- read.csv("{metadata_path}", row.names=1)
meta${design_col} <- as.factor(meta${design_col})
'''

        if reference_level:
            r_code += f'meta${design_col} <- relevel(meta${design_col}, ref="{reference_level}")\n'

        r_code += f'''
dds <- DESeqDataSetFromMatrix(countData=cts, colData=meta, design=as.formula(paste0("~", "{design_col}")))
dds <- DESeq(dds)
res <- results(dds, alpha={padj_threshold})
res_df <- as.data.frame(res[order(res$padj), ])
sig <- res_df[!is.na(res_df$padj) & res_df$padj < {padj_threshold}, ]

result <- list(
    total_genes = nrow(res_df),
    sig_genes = nrow(sig),
    padj_threshold = {padj_threshold},
    top_genes = head(res_df[, c("baseMean", "log2FoldChange", "pvalue", "padj")], 30)
)
cat("\\n---RESULT---\\n")
cat(jsonlite::toJSON(result, auto_unbox=TRUE, pretty=TRUE, digits=6))
'''

        res = _run_r_script(r_code, timeout=300)
        return json.dumps(res, ensure_ascii=False)
