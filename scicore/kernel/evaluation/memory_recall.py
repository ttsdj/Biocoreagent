"""Leakage-aware evaluation for cross-session memory retrieval."""

from __future__ import annotations

import statistics


def evaluate_memory_recall(cases, retrieve, *, k=10):
    """Evaluate a retriever against frozen, source-labelled query cases.

    Each case requires ``query`` and ``relevant_memory_ids``.  Optional scope
    fields are passed to the retriever, so tests can detect cross-user/project
    leakage instead of accidentally rewarding it.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("memory recall evaluation requires at least one case")
    reciprocal_ranks = []
    hits = 0
    leakage = 0
    details = []
    for case in cases:
        expected = {str(item) for item in case["relevant_memory_ids"]}
        results = list(
            retrieve(
                str(case["query"]),
                limit=int(k),
                user_scope=str(case.get("user_scope", "")),
                project_scope=str(case.get("project_scope", "")),
            )
        )[: int(k)]
        rank = next(
            (index + 1 for index, item in enumerate(results) if str(item.get("memory_id", "")) in expected),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        forbidden = set(str(item) for item in case.get("forbidden_memory_ids", []))
        leaked = [str(item.get("memory_id", "")) for item in results if str(item.get("memory_id", "")) in forbidden]
        leakage += int(bool(leaked))
        details.append(
            {
                "case_id": str(case.get("case_id", "")),
                "first_relevant_rank": rank,
                "hit_at_k": rank is not None,
                "leaked_memory_ids": leaked,
                "returned_memory_ids": [str(item.get("memory_id", "")) for item in results],
            }
        )
    total = len(cases)
    return {
        "query_count": total,
        f"hit_rate_at_{k}": hits / total,
        f"mrr_at_{k}": statistics.mean(reciprocal_ranks),
        "scope_leakage_rate": leakage / total,
        "details": details,
        "interpretation": (
            "HitRate measures whether any labelled evidence appeared in the top-k; "
            "MRR measures the rank of the first labelled evidence.  Values are "
            "only valid for the supplied frozen cases."
        ),
    }
