"""Small TSV helpers for lightweight bioinformatics tools."""

from __future__ import annotations

import csv
from pathlib import Path


def read_tsv(path: str) -> list[dict[str, str]]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_count_matrix(path: str) -> tuple[list[str], list[dict[str, float]]]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError("count matrix must have a gene column and at least one sample column")
        gene_col = reader.fieldnames[0]
        sample_cols = reader.fieldnames[1:]
        rows = []
        for line_no, row in enumerate(reader, start=2):
            gene = row.get(gene_col, "").strip()
            if not gene:
                raise ValueError(f"missing gene id at line {line_no}")
            parsed = {"gene": gene}
            for sample in sample_cols:
                value = row.get(sample, "")
                try:
                    parsed[sample] = float(value)
                except ValueError as e:
                    raise ValueError(f"non-numeric count for sample {sample!r} at line {line_no}: {value!r}") from e
            rows.append(parsed)
    return sample_cols, rows


def file_error(path: str) -> str | None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: {path} not found"
    if not p.is_file():
        return f"Error: {path} is a directory, not a file"
    return None
