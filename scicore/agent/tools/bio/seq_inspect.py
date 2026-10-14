"""Safe FASTA/FASTQ metadata inspection."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from ..base import Tool


class BioSeqInspectTool(Tool):
    name = "bio_seq_inspect"
    description = (
        "Inspect FASTA or FASTQ files and return file-level sequence metadata. "
        "Does not print complete sequences or infer biological conclusions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to FASTA/FASTQ file"},
            "format": {
                "type": "string",
                "enum": ["fasta", "fastq"],
                "description": "Input format: fasta or fastq",
            },
            "max_records": {
                "type": "integer",
                "description": "Maximum records to scan. Default 10000.",
            },
        },
        "required": ["file_path", "format"],
    }

    def execute(self, file_path: str, format: str, max_records: int = 10000) -> str:
        fmt = format.lower()
        if fmt not in {"fasta", "fastq"}:
            return "Error: unsupported format. Supported formats: fasta, fastq"
        if max_records <= 0:
            return "Error: max_records must be positive"

        try:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            lengths: list[int] = []
            first_record_id = None
            truncated = False
            for record_id, seq_len in _iter_records(p, fmt):
                if first_record_id is None:
                    first_record_id = record_id
                if len(lengths) >= max_records:
                    truncated = True
                    break
                lengths.append(seq_len)

            result = {
                "file_path": str(p),
                "format": fmt,
                "record_count_seen": len(lengths),
                "total_bases_seen": sum(lengths),
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "mean_length": round(mean(lengths), 2) if lengths else 0,
                "first_record_id": first_record_id,
                "truncated": truncated,
                "notes": [
                    "Only metadata is reported; complete sequences are not printed.",
                    "No biological interpretation was generated.",
                ],
            }
            return json.dumps(result, indent=2)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"


def _iter_records(path: Path, fmt: str):
    try:
        from Bio import SeqIO  # type: ignore

        for record in SeqIO.parse(str(path), fmt):
            yield str(record.id), len(record.seq)
        return
    except ImportError:
        pass

    if fmt == "fasta":
        yield from _iter_fasta_basic(path)
    else:
        yield from _iter_fastq_basic(path)


def _iter_fasta_basic(path: Path):
    current_id = None
    length = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, length
                current_id = line[1:].split()[0] or "(blank-id)"
                length = 0
            else:
                if current_id is None:
                    raise ValueError("invalid FASTA: sequence data before first header")
                length += len(line)
    if current_id is not None:
        yield current_id, length


def _iter_fastq_basic(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline()
            plus = f.readline()
            qual = f.readline()
            if not (seq and plus and qual):
                raise ValueError("invalid FASTQ: incomplete four-line record")
            if not header.startswith("@") or not plus.startswith("+"):
                raise ValueError("invalid FASTQ: expected @ header and + separator")
            yield header[1:].strip().split()[0] or "(blank-id)", len(seq.strip())
