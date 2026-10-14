import time

from scicore.kernel.features.local_working_memory import LocalWorkingMemoryStore
from scicore.kernel.memory_pipeline import EvidenceDistiller, MemoryPipeline
from scicore.kernel.session_store import SessionStore


class _CaptureBackend:
    def __init__(self):
        self.items = []

    def upsert(self, item):
        self.items.append(item)
        return item["memory_id"]


def test_session_chunks_are_immutable_and_source_addressable(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "session-memory-1",
        "created_at": "2026-07-30T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "memory": {},
        "history": [
            {"role": "user", "content": "请始终用中文输出报告。", "created_at": "2026-07-30T00:01:00+00:00"},
            {"role": "assistant", "content": "决定采用 SQLite 会话态和 Postgres 长期记忆。", "created_at": "2026-07-30T00:02:00+00:00"},
        ],
    }
    store.save(session)

    chunks = store.create_chunks(session["id"], max_chars=120)

    assert chunks
    assert chunks[0]["source_uri"].startswith("session://session-memory-1/messages/")
    assert len(chunks[0]["content_sha256"]) == 64
    excerpt = store.source_excerpt(session["id"], 0, 1)
    assert [item["content"] for item in excerpt] == [
        "请始终用中文输出报告。",
        "决定采用 SQLite 会话态和 Postgres 长期记忆。",
    ]


def test_memory_pipeline_queues_then_distills_with_source_evidence(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "session-memory-2",
        "created_at": "2026-07-30T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "memory": {},
        "history": [
            {"role": "user", "content": "我希望报告使用中文。", "created_at": "2026-07-30T00:01:00+00:00"},
            {"role": "assistant", "content": "决定使用可审计的证据回跳。", "created_at": "2026-07-30T00:02:00+00:00"},
        ],
    }
    store.save(session)
    backend = _CaptureBackend()
    pipeline = MemoryPipeline(store, backend=backend)

    assert pipeline.capture(session, metadata={"user_scope": "u-1", "project_scope": "p-1"}) == 1
    result = pipeline.drain()

    assert result["processed"] == 1
    assert len(backend.items) == 2
    assert all(item["evidence"]["source_uri"].startswith("session://") for item in backend.items)
    assert {item["memory_type"] for item in backend.items} == {"user_preference", "decision"}
    assert store.pending_distillations() == []
    local = pipeline.local_store.retrieve("报告 中文", user_scope="u-1", project_scope="p-1")
    assert local and local[0]["kind"] == "local_working"


def test_memory_pipeline_keeps_outbox_when_postgres_is_not_configured(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "session-memory-3",
        "created_at": "2026-07-30T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "memory": {},
        "history": [{"role": "user", "content": "请用中文。", "created_at": "2026-07-30T00:01:00+00:00"}],
    }
    store.save(session)
    pipeline = MemoryPipeline(store, backend=None)

    pipeline.capture(session)
    result = pipeline.drain()

    assert result["processed"] == 0
    assert result["queued"] == 1


def test_distillation_identity_deduplicates_chunk_boundary_changes():
    distiller = EvidenceDistiller()
    first = distiller.distill(
        {
            "metadata": {"user_scope": "u", "project_scope": "p"},
            "chunk": {
                "content": "[assistant] 决定使用证据回跳。",
                "source_uri": "session://one/messages/1-1#sha256=a",
                "content_sha256": "a",
                "session_id": "one",
                "start_message_index": 1,
                "end_message_index": 1,
            },
        }
    )
    later = distiller.distill(
        {
            "metadata": {"user_scope": "u", "project_scope": "p"},
            "chunk": {
                "content": "[assistant] 决定使用证据回跳。\n[tool] more output",
                "source_uri": "session://one/messages/1-2#sha256=b",
                "content_sha256": "b",
                "session_id": "one",
                "start_message_index": 1,
                "end_message_index": 2,
            },
        }
    )

    assert first[0]["memory_id"] == later[0]["memory_id"]
    assert first[0]["evidence"]["source_uri"] != later[0]["evidence"]["source_uri"]


def test_memory_source_cli_returns_original_messages(tmp_path, capsys):
    from scicore.bio.cli import main

    store = SessionStore(tmp_path / ".scicore" / "sessions")
    store.save(
        {
            "id": "source-cli-session",
            "created_at": "2026-07-30T00:00:00+00:00",
            "workspace_root": str(tmp_path),
            "memory": {},
            "history": [{"role": "user", "content": "原始输入", "created_at": "2026-07-30T00:01:00+00:00"}],
        }
    )

    assert main(["--cwd", str(tmp_path), "--memory-source", "source-cli-session", "0", "0"]) == 0
    assert "原始输入" in capsys.readouterr().out


def test_local_working_memory_keeps_user_and_project_markdown_mirrors(tmp_path):
    store = LocalWorkingMemoryStore(tmp_path / ".scicore" / "memory")
    store.upsert(
        {
            "memory_type": "user_preference",
            "statement": "Reports should be written in Chinese.",
            "tags": ["preference"],
            "user_scope": "user-a",
            "project_scope": "project-a",
            "evidence": {"source_uri": "session://one/messages/0-0", "source_sha256": "d" * 64},
        }
    )
    store.upsert(
        {
            "memory_type": "decision",
            "statement": "The project uses evidence backlinks.",
            "tags": ["decision"],
            "user_scope": "user-a",
            "project_scope": "project-a",
            "evidence": {"source_uri": "session://one/messages/1-1", "source_sha256": "e" * 64},
        }
    )

    assert store.retrieve("Chinese reports", user_scope="user-a", project_scope="project-a")
    assert (tmp_path / ".scicore" / "memory" / "user" / "user-a.md").exists()
    assert (tmp_path / ".scicore" / "memory" / "project" / "project-a.md").exists()


def test_memory_outbox_can_drain_in_background_without_blocking_delivery(tmp_path):
    class _SlowBackend(_CaptureBackend):
        def upsert(self, item):
            time.sleep(0.3)
            return super().upsert(item)

    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "session-memory-async",
        "created_at": "2026-07-30T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "memory": {},
        "history": [
            {"role": "user", "content": "我希望所有报告都使用中文输出。"},
        ],
    }
    store.save(session)
    pipeline = MemoryPipeline(store, backend=_SlowBackend())
    pipeline.capture(session)

    started = time.monotonic()
    status = pipeline.drain_async()
    elapsed = time.monotonic() - started

    assert status["status"] == "running"
    assert elapsed < 0.15
    assert pipeline.close(timeout=5)["status"] == "completed"
