"""Task routing for SciCoreAgent analysis workflows.

The router deliberately uses simple deterministic rules. It is not meant to
replace the model; it decides when a request is risky enough to leave the free
tool loop and enter a governed analysis path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class AnalysisRoute:
    analysis_type: str
    intent: str
    risk_level: str
    requires_plan: bool
    preferred_backend: str
    fallback_allowed: bool
    input_path: str = ""
    target_group: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def route_analysis_task(text: str) -> AnalysisRoute:
    request = str(text or "")
    lowered = request.lower()
    path = extract_table_path(request)
    inspect_only = _is_inspect_only(lowered)

    if _has_any(lowered, ("proteomics", "protein intensity", "lfq", "tmt", "itraq", "蛋白组", "蛋白質组", "蛋白质组")):
        return AnalysisRoute(
            analysis_type="proteomics",
            intent="inspect" if inspect_only else "run_analysis",
            risk_level="medium",
            requires_plan=not inspect_only,
            preferred_backend="generic_script",
            fallback_allowed=True,
            input_path=path,
        )
    if _has_any(lowered, ("scrna", "single-cell", "single cell", "h5ad", "10x", "单细胞", "單細胞")):
        return AnalysisRoute(
            analysis_type="single_cell",
            intent="inspect" if inspect_only else "run_analysis",
            risk_level="medium",
            requires_plan=not inspect_only,
            preferred_backend="generic_script",
            fallback_allowed=True,
            input_path=path,
        )
    if _is_bulk_rnaseq(lowered):
        return AnalysisRoute(
            analysis_type="bulk_rnaseq",
            intent="inspect" if inspect_only else "run_analysis",
            risk_level="medium",
            requires_plan=not inspect_only,
            preferred_backend="omicverse",
            fallback_allowed=True,
            input_path=path,
            target_group=extract_target_group(request),
        )
    if path and _has_any(lowered, ("analyze", "analysis", "compare", "run", "分析", "比较", "差异", "整理")):
        return AnalysisRoute(
            analysis_type="generic_table",
            intent="run_analysis",
            risk_level="medium",
            requires_plan=False,
            preferred_backend="generic_script",
            fallback_allowed=True,
            input_path=path,
        )
    return AnalysisRoute(
        analysis_type="coding",
        intent="inspect" if inspect_only else "run_analysis",
        risk_level="low",
        requires_plan=False,
        preferred_backend="model_loop",
        fallback_allowed=False,
        input_path=path,
    )


def extract_table_path(text: str) -> str:
    extensions = r"(?:txt|tsv|csv|xlsx|h5ad|mtx)"
    quoted = re.findall(rf'"([^"]+\.{extensions})"', text, flags=re.I)
    if quoted:
        return quoted[0]
    absolute = re.findall(rf"([A-Za-z]:\\[^\s\"']+\.{extensions})", text, flags=re.I)
    if absolute:
        return absolute[0]
    relative = re.findall(rf"([^\s\"']+\.{extensions})", text, flags=re.I)
    return relative[0] if relative else ""


def extract_target_group(text: str) -> str:
    patterns = [
        r"计算\s*([A-Za-z]{1,20})\s*与",
        r"([A-Za-z]{1,20})\s*与其余",
        r"([A-Za-z]{1,20})\s*和其余",
        r"([A-Za-z]{1,20})\s*对其余",
        r"([A-Za-z]{1,20})\s*(?:vs|versus)\s*(?:rest|others)",
        r"target[_\s-]*group[:=]\s*([A-Za-z]{1,20})",
        r"target[_\s-]*tissue[:=]\s*([A-Za-z]{1,20})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).lower()
    return ""


def infer_target_from_matrix(path: str, text: str) -> str:
    try:
        matrix = Path(path).resolve()
        header = matrix.read_text(encoding="utf-8-sig").splitlines()[0]
    except Exception:
        return ""
    delimiter = "," if matrix.suffix.lower() == ".csv" else "\t"
    columns = header.split(delimiter)
    prefixes = set()
    for sample in columns[1:]:
        match = re.match(r"([A-Za-z]+)", sample.strip())
        if match:
            prefixes.add(match.group(1).lower())
    lowered = str(text).lower()
    candidates = [prefix for prefix in sorted(prefixes, key=len, reverse=True) if re.search(rf"\b{re.escape(prefix)}\b", lowered)]
    return candidates[0] if len(candidates) == 1 else ""


def _is_bulk_rnaseq(lowered: str) -> bool:
    strong = (
        "deseq2",
        "rna-seq",
        "rnaseq",
        "bulk rna",
        "转录组",
        "轉錄組",
        "差异表达",
        "差異表達",
        "表达矩阵",
        "基因表达",
    )
    if _has_any(lowered, strong):
        return True
    count_terms = ("count matrix", "counts", "count矩阵", "count")
    actions = ("分析", "比较", "計算", "计算", "差异", "运行", "run", "compare", "vs", "与其余")
    return _has_any(lowered, count_terms) and _has_any(lowered, actions)


def _is_inspect_only(lowered: str) -> bool:
    if _has_any(lowered, ("inspect", "检查", "查看", "读取")):
        return not _has_any(lowered, ("分析", "比较", "差异", "deseq2", "run", "compare", "vs", "与其余"))
    return False


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
