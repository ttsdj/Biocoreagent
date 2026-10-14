"""State tracker for bulk transcriptome inputs and artifacts."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptomeState:
    count_matrix_path: str = ""
    metadata_path: str = ""
    sample_columns: list[str] = field(default_factory=list)
    sample_groups: dict[str, int] = field(default_factory=dict)
    target_group: str = ""
    target_replicates: int = 0
    rest_replicates: int = 0
    gene_rows_seen: int = 0
    integer_like_counts: bool = False
    count_matrix_inspected: bool = False
    target_has_replicates: bool = False
    rest_has_replicates: bool = False
    deseq2_environment_checked: bool = False
    omicverse_backend_checked: bool = False
    differential_expression_completed: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_transcriptome_state(
    count_matrix_path: str = "",
    metadata_path: str = "",
    target_group: str = "",
    max_rows: int = 10000,
) -> TranscriptomeState:
    state = TranscriptomeState(
        count_matrix_path=str(count_matrix_path or ""),
        metadata_path=str(metadata_path or ""),
        target_group=str(target_group or "").strip().lower(),
    )
    if not count_matrix_path:
        state.blockers.append("missing_count_matrix_path")
        return state
    path = Path(count_matrix_path).expanduser().resolve()
    state.count_matrix_path = str(path)
    if not path.exists():
        state.blockers.append("count_matrix_not_found")
        return state
    if not path.is_file():
        state.blockers.append("count_matrix_is_not_file")
        return state

    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    groups: Counter[str] = Counter()
    non_integer = 0
    rows_seen = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        state.sample_columns = [col.strip() for col in header[1:] if col.strip()]
        if len(state.sample_columns) < 2:
            state.blockers.append("count_matrix_requires_at_least_two_samples")
        for sample in state.sample_columns:
            match = re.match(r"([A-Za-z]+)", sample)
            if match:
                groups[match.group(1).lower()] += 1
            else:
                state.warnings.append(f"sample_group_not_inferred:{sample}")
        for rows_seen, row in enumerate(reader, start=1):
            if rows_seen > max_rows:
                break
            for value in row[1:]:
                try:
                    numeric = float(value)
                    if numeric != int(numeric):
                        non_integer += 1
                except ValueError:
                    non_integer += 1
    state.gene_rows_seen = min(rows_seen, max_rows)
    state.sample_groups = dict(groups)
    state.integer_like_counts = non_integer == 0
    state.count_matrix_inspected = bool(state.sample_columns) and state.integer_like_counts and not any(
        blocker.startswith("count_matrix_") for blocker in state.blockers
    )
    if not state.integer_like_counts:
        state.blockers.append("count_matrix_has_non_integer_or_non_numeric_values")
    if state.target_group:
        state.target_replicates = groups.get(state.target_group, 0)
        state.rest_replicates = sum(groups.values()) - state.target_replicates
        state.target_has_replicates = state.target_replicates >= 2
        state.rest_has_replicates = state.rest_replicates >= 2
        if not state.target_has_replicates:
            state.blockers.append("target_group_requires_at_least_two_replicates")
        if not state.rest_has_replicates:
            state.blockers.append("rest_group_requires_at_least_two_samples")
    elif groups:
        state.warnings.append("target_group_not_specified")
    if metadata_path:
        meta = Path(metadata_path).expanduser().resolve()
        state.metadata_path = str(meta)
        if not meta.exists():
            state.blockers.append("metadata_not_found")
    return state
