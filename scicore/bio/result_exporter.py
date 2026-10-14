"""Deterministic result-table export helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass(frozen=True)
class CsvExportResult:
    status: str
    source_path: str
    output_path: str
    rows: int = 0
    columns: int = 0
    delimiter: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def is_csv_export_request(text: str) -> bool:
    lowered = _strip_path_like_text(str(text or "")).lower()
    wants_csv = "csv" in lowered
    action = any(term in lowered for term in ("export", "convert", "save as", "output", "输出", "导出", "轉成", "转成", "转换", "另存"))
    return wants_csv and action


def export_result_table_to_csv(root: Path, request: str) -> CsvExportResult:
    source = _extract_source_path(root, request) or _find_likely_result_table(root)
    if source is None:
        return CsvExportResult(
            status="failed",
            source_path="",
            output_path="",
            message="No result table was found in the current workspace.",
        )
    if not source.exists():
        return CsvExportResult(
            status="failed",
            source_path=str(source),
            output_path="",
            message=f"Source table does not exist: {source}",
        )
    output = _non_overwriting_csv_path(source)
    delimiter = _detect_delimiter(source)
    rows = 0
    columns = 0
    with source.open("r", encoding="utf-8-sig", newline="") as handle, output.open("w", encoding="utf-8-sig", newline="") as out:
        reader = csv.reader(handle, delimiter=delimiter)
        writer = csv.writer(out)
        for row in reader:
            if rows == 0:
                columns = len(row)
            writer.writerow(row)
            rows += 1
    return CsvExportResult(
        status="completed",
        source_path=str(source),
        output_path=str(output),
        rows=max(rows - 1, 0),
        columns=columns,
        delimiter="tab" if delimiter == "\t" else delimiter,
        message="CSV export completed.",
    )


def _extract_source_path(root: Path, text: str) -> Path | None:
    extensions = r"(?:txt|tsv|csv)"
    quoted = re.findall(rf'"([^"]+\.{extensions})"', text, flags=re.I)
    candidates = quoted or re.findall(rf"([A-Za-z]:\\[^\s\"']+\.{extensions})", text, flags=re.I)
    if not candidates:
        candidates = re.findall(rf"([^\s\"']+\.{extensions})", text, flags=re.I)
    if not candidates:
        return None
    path = Path(candidates[0])
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _strip_path_like_text(text: str) -> str:
    text = re.sub(r'"[^"]+\.(?:txt|tsv|csv)"', " ", text, flags=re.I)
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+\.(?:txt|tsv|csv)", " ", text, flags=re.I)
    text = re.sub(r"[^\s\"']+\.(?:txt|tsv|csv)", " ", text, flags=re.I)
    return text


def _find_likely_result_table(root: Path) -> Path | None:
    patterns = ("*results.txt", "*result.txt", "*results.tsv", "*result.tsv", "*.txt", "*.tsv")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        priority = 0
        if "deseq2" in name:
            priority += 30
        if "result" in name:
            priority += 20
        if "significant" in name:
            priority += 5
        return (priority, path.stat().st_mtime)

    return sorted(candidates, key=score, reverse=True)[0]


def _non_overwriting_csv_path(source: Path) -> Path:
    base = source.with_suffix(".csv")
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return source.with_name(f"{source.stem}_{stamp}.csv")


def _detect_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    if "\t" in sample:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except Exception:
        return "\t"
