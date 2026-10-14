"""Optional OmicVerse backend for bulk transcriptome capabilities."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any


def check_omicverse_bulk_backend() -> dict[str, Any]:
    spec = importlib.util.find_spec("omicverse")
    if spec is None:
        return {
            "available": False,
            "status": "backend_missing",
            "backend": "omicverse",
            "message": "omicverse is not installed. Install it to enable OmicVerse-backed transcriptome execution.",
        }
    try:
        ov = importlib.import_module("omicverse")
        version = getattr(ov, "__version__", "")
        bulk = getattr(ov, "bulk", None)
        required = ["pyDEG", "deseq2_normalize", "Matrix_ID_mapping"]
        missing = [name for name in required if bulk is None or not hasattr(bulk, name)]
        return {
            "available": not missing,
            "status": "ok" if not missing else "missing_bulk_symbols",
            "backend": "omicverse",
            "version": version,
            "missing_symbols": missing,
            "required_symbols": required,
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "backend_import_error",
            "backend": "omicverse",
            "error": str(exc),
        }


def run_omicverse_pydge_tissue_vs_rest(
    count_matrix_path: str,
    target_group: str,
    output_dir: str = "",
    method: str = "DEseq2",
    alpha: float = 0.05,
    n_cpus: int = 2,
) -> dict[str, Any]:
    backend = check_omicverse_bulk_backend()
    if not backend.get("available"):
        return {
            "status": "backend_missing",
            "backend_check": backend,
            "fallback_hint": "Use bio_deseq2_tissue_vs_rest or install omicverse with compatible dependencies.",
        }

    import pandas as pd
    import omicverse as ov

    matrix = Path(count_matrix_path).expanduser().resolve()
    if not matrix.exists():
        return {"status": "error", "error": f"count matrix not found: {matrix}"}
    target = str(target_group or "").strip().lower()
    if not target:
        return {"status": "error", "error": "target_group is required"}
    out = Path(output_dir).expanduser().resolve() if output_dir else matrix.parent
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(matrix, sep="," if matrix.suffix.lower() == ".csv" else "\t", index_col=0)
    samples = [str(col) for col in data.columns]
    groups = [_sample_prefix(sample) for sample in samples]
    treatment = [sample for sample, group in zip(samples, groups) if group == target]
    control = [sample for sample, group in zip(samples, groups) if group != target]
    if len(treatment) < 2:
        return {"status": "error", "error": "target group requires at least two replicates", "target_samples": treatment}
    if len(control) < 2:
        return {"status": "error", "error": "rest group requires at least two samples", "rest_samples": control}

    prefix = f"omicverse_{target}_vs_rest"
    full_csv = out / f"{prefix}_full_results.csv"
    sig_csv = out / f"{prefix}_significant.csv"
    summary_json = out / f"{prefix}_summary.json"

    deg = ov.bulk.pyDEG(data)
    deg.drop_duplicates_index()
    result = deg.deg_analysis(treatment, control, method=method, alpha=float(alpha), n_cpus=int(n_cpus or 2))
    if "qvalue" in result.columns:
        sig = result.loc[result["qvalue"] < float(alpha)].copy()
    elif "padj" in result.columns:
        sig = result.loc[result["padj"] < float(alpha)].copy()
    else:
        sig = result.iloc[0:0].copy()
    result.to_csv(full_csv)
    sig.to_csv(sig_csv)
    summary = {
        "backend": "omicverse",
        "method": method,
        "target_group": target,
        "target_samples": treatment,
        "rest_sample_count": len(control),
        "total_genes": int(result.shape[0]),
        "significant_genes": int(sig.shape[0]),
        "up_in_target": int((sig.get("log2FC", 0) > 0).sum()) if "log2FC" in sig else 0,
        "down_in_target": int((sig.get("log2FC", 0) < 0).sum()) if "log2FC" in sig else 0,
        "full_results_path": str(full_csv),
        "significant_results_path": str(sig_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "completed",
        "backend_check": backend,
        "summary": summary,
        "summary_path": str(summary_json),
        "full_results_path": str(full_csv),
        "significant_results_path": str(sig_csv),
    }


def _sample_prefix(sample: str) -> str:
    match = re.match(r"([A-Za-z]+)", sample.strip())
    return match.group(1).lower() if match else sample.lower()
