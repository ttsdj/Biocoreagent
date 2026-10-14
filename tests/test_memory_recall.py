from scicore.kernel.evaluation.memory_recall import evaluate_memory_recall


def test_memory_recall_reports_hitrate_mrr_and_scope_leakage():
    records = {
        "rna": {"memory_id": "rna", "text": "DESeq2 requires metadata"},
        "private": {"memory_id": "private", "text": "another user's preference"},
    }

    def retrieve(query, *, limit, user_scope, project_scope):
        assert user_scope == "u-1"
        assert project_scope == "p-1"
        return [records["private"], records["rna"]]

    result = evaluate_memory_recall(
        [
            {
                "case_id": "q-1",
                "query": "RNA metadata",
                "relevant_memory_ids": ["rna"],
                "forbidden_memory_ids": ["private"],
                "user_scope": "u-1",
                "project_scope": "p-1",
            }
        ],
        retrieve,
        k=10,
    )

    assert result["hit_rate_at_10"] == 1.0
    assert result["mrr_at_10"] == 0.5
    assert result["scope_leakage_rate"] == 1.0
