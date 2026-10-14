import sqlite3

from scicore.kernel.context_manager import ContextManager
from scicore.kernel.features.memory import LayeredMemory
from scicore.kernel.session_store import SessionStore
from scicore.agent.sedimentation import DREAM_STAGES, DreamSedimentationManager, SedimentationReviewer
from scicore.agent.skills import read_skill


def test_session_store_uses_sqlite_and_keeps_json_mirror(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "session-1",
        "created_at": "2026-07-24T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "history": [],
        "memory": {},
    }

    mirror_path = store.save(session)

    assert store.db_path.exists()
    assert mirror_path.exists()
    assert store.load("session-1") == session
    assert store.latest() == "session-1"
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1


def test_durable_memory_hybrid_retrieval_returns_source_backlink(tmp_path):
    memory = LayeredMemory(workspace_root=tmp_path)
    promoted, superseded = memory.promote_durable(
        [
            ("project-conventions", "Bulk RNA-seq comparisons require explicit sample metadata."),
            ("dependency-facts", "DESeq2 requires a compatible rlang package version."),
            ("user-preferences", "Reports should be written in Chinese."),
        ]
    )

    results = memory.retrieval_candidates("RNA seq sample metadata", limit=3)

    assert promoted
    assert superseded == []
    assert results[0]["text"].startswith("Bulk RNA-seq")
    assert results[0]["retrieval"]["method"] == "rrf_bm25_dense_hash"
    assert results[0]["source"].endswith("project-conventions.md")
    assert results[0]["source_anchor"].startswith("sha256:")
    assert (tmp_path / ".scicore/kernel" / "memory" / "memory.sqlite").exists()


class _ContextAgent:
    def __init__(self, tmp_path):
        self.prefix = "system rules " + ("P" * 400)
        self.session = {"history": []}
        self.memory = LayeredMemory(workspace_root=tmp_path)

    def memory_text(self):
        return self.memory.render_memory_text()

    def feature_enabled(self, name):
        return True


def test_context_metadata_exposes_four_layer_compression_pipeline(tmp_path):
    agent = _ContextAgent(tmp_path)
    agent.memory.set_file_summary("large.txt", "large file summary")
    agent.memory.remember_file("large.txt")
    for index in range(10):
        agent.session["history"].append(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": "large.txt"},
                "content": f"payload-{index}-" + ("X" * 500),
            }
        )
    manager = ContextManager(
        agent,
        total_budget=650,
        section_budgets={
            "prefix": 260,
            "memory": 160,
            "relevant_memory": 120,
            "history": 300,
        },
    )

    prompt, metadata = manager.build("summarize large.txt")

    assert len(prompt) <= 650
    assert metadata["raw_prompt_chars"] > metadata["compressed_chars"]
    assert 0 < metadata["compression_ratio"] < 1
    assert [item["name"] for item in metadata["compression_layers"]] == [
        "budget_clip",
        "redundancy_prune",
        "structured_compact",
        "adaptive_threshold_trim",
    ]
    assert metadata["history"]["collapsed_duplicate_reads"] > 0
    assert sum(item["removed_chars"] for item in metadata["compression_layers"]) == metadata["removed_chars"]


def test_background_review_creates_and_versions_draft_skill(tmp_path):
    reviewer = SedimentationReviewer(tmp_path, threshold=3)
    tools = [{"name": "bio_count_matrix_inspect"}, {"name": "bio_rnaseq_compare"}]

    assert reviewer.review("RNA-seq run 1", "completed 1", tools) == []
    assert reviewer.review("RNA-seq run 2", "completed 2", tools) == []
    created = reviewer.review("RNA-seq run 3", "completed 3", tools)

    assert created[0]["action"] == "created"
    assert created[0]["version"] == 1
    assert "tags: auto, draft, reviewed, bulk-rnaseq" in read_skill(
        "learned-bulk-rnaseq",
        tmp_path,
    )

    assert reviewer.review("RNA-seq run 4", "completed 4", tools) == []
    assert reviewer.review("RNA-seq run 5", "completed 5", tools) == []
    updated = reviewer.review("RNA-seq run 6", "completed 6", tools)

    assert updated[0]["action"] == "updated"
    assert updated[0]["version"] == 2
    assert list(
        (tmp_path / ".scicore" / "skills" / ".history" / "learned-bulk-rnaseq").glob("v1_*.md")
    )
    assert (tmp_path / ".scicore" / "wiki" / "entries.jsonl").exists()


def test_background_review_does_not_learn_from_failure(tmp_path):
    reviewer = SedimentationReviewer(tmp_path, threshold=3)

    result = reviewer.review(
        "RNA-seq failed run",
        "分析未完成：DESeq2 execution failed",
        [{"name": "bio_rnaseq_compare"}],
    )

    assert result == []
    assert not reviewer.observations_path.exists()


def test_dream_queue_is_async_durable_and_tracks_five_stages(tmp_path):
    manager = DreamSedimentationManager(tmp_path, threshold=3)
    tools = [{"name": "bio_count_matrix_inspect", "metadata": {"tool_status": "success"}}]
    try:
        submitted = manager.submit(
            "RNA-seq asynchronous review",
            "completed with verified artifact",
            tools,
            source_session="session-dream",
        )

        assert submitted["status"] == "queued"
        assert tuple(submitted["stages"]) == DREAM_STAGES
        completed = manager.wait(submitted["job_id"], timeout=5)
        assert completed["status"] == "completed"
        assert set(completed["stages"].values()) == {"completed"}
        assert (
            tmp_path
            / ".scicore"
            / "sedimentation"
            / "dream"
            / "jobs"
            / f"{submitted['job_id']}.json"
        ).exists()
    finally:
        manager.close()
