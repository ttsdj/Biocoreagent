"""Controlled fallback script runner for unregistered analysis types."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from datetime import datetime


@dataclass
class FallbackResult:
    status: str
    analysis_type: str
    input_path: str
    work_dir: str
    script_path: str
    summary_path: str
    stdout_path: str
    stderr_path: str
    error_type: str = ""
    blocker: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_fallback_analysis(root: Path, analysis_type: str, input_path: str, timeout: int = 120) -> FallbackResult:
    source = Path(input_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = root / f"{analysis_type}_fallback_{stamp}"
    outputs = work_dir / "outputs"
    logs = work_dir / "logs"
    outputs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    script = work_dir / "analysis_script.py"
    summary = work_dir / "summary.json"
    stdout_path = logs / "run_1.stdout.txt"
    stderr_path = logs / "run_1.stderr.txt"
    repair_log = logs / "repair_log.json"

    script.write_text(_render_table_summary_script(source, summary), encoding="utf-8")
    if not source.exists():
        result = FallbackResult(
            status="failed",
            analysis_type=analysis_type,
            input_path=str(source),
            work_dir=str(work_dir),
            script_path=str(script),
            summary_path=str(summary),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error_type="file_not_found",
            blocker=f"Input file not found: {source}",
        )
        repair_log.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    try:
        proc = subprocess.run(
            ["python", str(script)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or f"timed out after {timeout} seconds", encoding="utf-8")
        result = FallbackResult(
            status="failed",
            analysis_type=analysis_type,
            input_path=str(source),
            work_dir=str(work_dir),
            script_path=str(script),
            summary_path=str(summary),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error_type="timeout",
            blocker=f"fallback script timed out after {timeout} seconds",
        )
        repair_log.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    status = "completed" if proc.returncode == 0 and summary.exists() else "failed"
    error_type = "" if status == "completed" else classify_error(proc.returncode, proc.stderr)
    blocker = "" if status == "completed" else (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}")
    result = FallbackResult(
        status=status,
        analysis_type=analysis_type,
        input_path=str(source),
        work_dir=str(work_dir),
        script_path=str(script),
        summary_path=str(summary),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        error_type=error_type,
        blocker=blocker,
    )
    repair_log.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def classify_error(returncode: int, stderr: str) -> str:
    lowered = str(stderr or "").lower()
    if returncode == 3221225477 or "0xc0000005" in lowered:
        return "native_crash"
    if "version" in lowered and ("requires" in lowered or ">=" in lowered or "too old" in lowered):
        return "package_version_too_low"
    if "no module named" in lowered or "there is no package" in lowered or "modulenotfounderror" in lowered:
        return "missing_package"
    if "syntaxerror" in lowered or "parse error" in lowered:
        return "script_syntax_error"
    if "expected" in lowered and ("columns" in lowered or "header" in lowered):
        return "data_schema_error"
    if "no such file" in lowered or "cannot open" in lowered:
        return "file_not_found"
    if "invalid argument" in lowered or "not a valid path" in lowered:
        return "path_error"
    if "timed out" in lowered:
        return "timeout"
    return "unknown_error"


def _render_table_summary_script(input_path: Path, summary_path: Path) -> str:
    return f'''import csv, json, math
from pathlib import Path

input_path = Path(r"{input_path}")
summary_path = Path(r"{summary_path}")
suffix = input_path.suffix.lower()

if suffix == ".xlsx":
    try:
        import openpyxl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("No module named 'openpyxl'; install openpyxl to read xlsx fallback inputs") from exc
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    ws = wb.active
    rows = [[cell for cell in row] for row in ws.iter_rows(values_only=True)]
else:
    delimiter = "," if suffix == ".csv" else "\\t"
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)

if not rows:
    raise SystemExit("empty input table")

header = [str(item) for item in rows[0]]
data = rows[1:]
numeric = {{}}
for idx, name in enumerate(header):
    values = []
    for row in data:
        if idx >= len(row):
            continue
        try:
            values.append(float(row[idx]))
        except Exception:
            pass
    if values:
        numeric[name] = {{
            "n": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }}

summary = {{
    "input_path": str(input_path),
    "rows": len(data),
    "columns": len(header),
    "column_names": header,
    "numeric_columns": numeric,
    "note": "Generic fallback summary only; domain-specific interpretation requires a registered capability.",
}}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
'''
