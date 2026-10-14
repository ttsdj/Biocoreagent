"""Runs only when a disposable Postgres DSN is explicitly configured."""

import os

import pytest

from scicore.kernel.features.postgres_memory import PostgresLongTermMemoryStore
from scicore.kernel.memory_pipeline import MemoryPipeline
from scicore.kernel.session_store import SessionStore


@pytest.mark.skipif(
    not os.environ.get("SCICORE_MEMORY_POSTGRES_DSN"),
    reason="requires SCICORE_MEMORY_POSTGRES_DSN",
)
def test_postgres_memory_hybrid_retrieval_and_scope_isolation():
    store = PostgresLongTermMemoryStore.from_environment()
    evidence = {
        "source_uri": "session://postgres-integration/messages/0-1#sha256=" + "b" * 64,
        "source_sha256": "b" * 64,
        "session_id": "postgres-integration",
        "start_message_index": 0,
        "end_message_index": 1,
    }
    expected = store.upsert(
        {
            "memory_type": "decision",
            "statement": "Postgres integration RNA metadata evidence marker alpha-7.",
            "tags": ["integration", "rna", "metadata"],
            "user_scope": "integration-user-a",
            "project_scope": "integration-project",
            "evidence": evidence,
        }
    )
    foreign = store.upsert(
        {
            "memory_type": "decision",
            "statement": "Postgres integration RNA metadata evidence marker alpha-7.",
            "tags": ["integration", "rna", "metadata"],
            "user_scope": "integration-user-b",
            "project_scope": "integration-project",
            "evidence": {**evidence, "source_uri": evidence["source_uri"].replace("0-1", "2-3")},
        }
    )

    rows = store.retrieve(
        "RNA metadata alpha-7",
        limit=10,
        user_scope="integration-user-a",
        project_scope="integration-project",
    )

    assert rows[0]["memory_id"] == expected
    assert foreign not in [row["memory_id"] for row in rows]
    assert rows[0]["retrieval"]["method"] == "rrf_dense_bm25"
    assert rows[0]["source"].startswith("session://postgres-integration/")


@pytest.mark.skipif(
    not os.environ.get("SCICORE_MEMORY_POSTGRES_DSN"),
    reason="requires SCICORE_MEMORY_POSTGRES_DSN",
)
def test_session_chunk_outbox_to_postgres_to_retrieval(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "postgres-pipeline-session",
        "created_at": "2026-07-30T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "memory": {},
        "history": [
            {
                "role": "user",
                "content": "Please always use report format marker pipeline-42.",
                "created_at": "2026-07-30T00:01:00+00:00",
            },
            {
                "role": "assistant",
                "content": "Decided use evidence backlinks for marker pipeline-42.",
                "created_at": "2026-07-30T00:02:00+00:00",
            },
        ],
    }
    store.save(session)
    backend = PostgresLongTermMemoryStore.from_environment()
    pipeline = MemoryPipeline(store, backend=backend)

    assert pipeline.capture(
        session,
        metadata={"user_scope": "pipeline-user", "project_scope": "pipeline-project"},
    ) == 1
    assert pipeline.drain()["processed"] == 1
    rows = backend.retrieve(
        "report format marker pipeline-42",
        limit=10,
        user_scope="pipeline-user",
        project_scope="pipeline-project",
    )

    assert rows
    assert rows[0]["source"].startswith("session://postgres-pipeline-session/messages/")
    assert rows[0]["retrieval"]["method"] == "rrf_dense_bm25"


@pytest.mark.skipif(
    not os.environ.get("SCICORE_MEMORY_POSTGRES_DSN"),
    reason="requires SCICORE_MEMORY_POSTGRES_DSN",
)
def test_postgres_research_graph_timeline_and_evidence():
    store = PostgresLongTermMemoryStore.from_environment()
    memory = {
        "memory_id": "mem-postgres-graph-integration",
        "memory_type": "decision",
        "statement": "决定使用 DESeq2 生成 graph_results.csv。",
        "user_scope": "graph-user",
        "project_scope": "graph-project",
        "created_at": "2026-07-30T08:00:00+00:00",
        "evidence": {
            "source_uri": "session://graph-integration/messages/0-1#sha256="
            + "c" * 64,
            "source_sha256": "c" * 64,
            "session_id": "graph-integration",
        },
    }

    event_id = store.upsert_graph_memory(memory)
    timeline = store.research_graph.timeline(
        user_scope="graph-user", project_scope="graph-project"
    )
    explained = store.research_graph.explain_event(event_id)

    assert event_id in [row["event_id"] for row in timeline]
    assert explained["evidence"][0]["source_uri"].startswith(
        "session://graph-integration/"
    )
