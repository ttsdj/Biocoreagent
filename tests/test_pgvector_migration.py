"""pgvector (C2) 可选向量检索迁移的集成测试.

仅在显式提供 ``BIOCOREAGENT_MEMORY_POSTGRES_DSN`` 时运行（与
``test_postgres_memory_integration.py`` 同一套 skipif 约定）。它假设该 DSN 指向
一个带 `vector` 扩展的 Postgres（仓库 docker-compose 的 ``memory-postgres`` 服务
用 ``pgvector/pgvector:pg15``，启动后即具备该扩展）。

覆盖三层，都围绕「pgvector 只缩短 dense 召回、scope 过滤仍先行、默认仍是 JSONB」：
  1. initialize() 幂等地加 `vector` 扩展 + `embedding vector(256)` 列 + HNSW，
     并把 ``pgvector_enabled`` 置为 True；
  2. upsert() 除了写 embedding_json（JSONB 兜底）外，还同步写 `embedding` 向量列；
  3. retrieve() 在 scope 过滤后的候选集上用 `embedding <=> query` 求 dense，
     再与 BM25 做 RRF 融合，仍返回 ``rrf_dense_bm25`` 的可寻址行。

无 DSN 时本文件整体跳过，**不**在仓库内伪造 pgvector 可用性。
"""

from __future__ import annotations

import os

import pytest

from pico.features.postgres_memory import PostgresLongTermMemoryStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("BIOCOREAGENT_MEMORY_POSTGRES_DSN"),
    reason="requires BIOCOREAGENT_MEMORY_POSTGRES_DSN",
)


def _evidence(session_id, left, right):
    digest = ("%02x" % left * 32) + ("%02x" % right * 32)  # 64 hex chars
    return {
        "source_uri": f"session://{session_id}/messages/{left}-{right}#sha256={digest}",
        "source_sha256": digest,
        "session_id": session_id,
        "start_message_index": left,
        "end_message_index": right,
    }


def test_pgvector_migration_applies_and_uses_native_dense():
    store = PostgresLongTermMemoryStore.from_environment()

    # 1) 迁移：扩展 + 向量列 + HNSW，且标志位为 True（针对 pgvector 镜像）。
    store.initialize()
    assert store.pgvector_enabled, (
        "pgvector not enabled — the DSN must point to a pgvector/pgvector image "
        "(docker-compose memory-postgres service)."
    )

    # 2) upsert 同步写向量列。
    memory_id = store.upsert(
        {
            "memory_type": "decision",
            "statement": "RNA-seq variability in pgvector migration marker epsilon-9.",
            "tags": ["pgvector", "migration", "rna"],
            "user_scope": "pgvector-user",
            "project_scope": "pgvector-project",
            "evidence": _evidence("pgvector-migration", 0, 1),
        }
    )

    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT embedding IS NOT NULL, embedding_model "
            "FROM long_term_memories WHERE memory_id = %s",
            (memory_id,),
        )
        has_vector, embedding_model = cursor.fetchone()
    assert has_vector is True  # 向量列被真实写入，而非依赖 JSONB 兜底
    assert embedding_model.startswith("dense_hash_fallback") or embedding_model.startswith(
        "sentence_transformers:"
    )

    # 3) retrieve：scope 过滤先行，dense 走 `<=>`，RRF 融合，仍可寻址。
    rows = store.retrieve(
        "RNA-seq variability epsilon-9",
        limit=10,
        user_scope="pgvector-user",
        project_scope="pgvector-project",
    )
    assert rows
    assert rows[0]["memory_id"] == memory_id
    assert rows[0]["retrieval"]["method"] == "rrf_dense_bm25"
    assert rows[0]["source"].startswith("session://pgvector-migration/")
    assert rows[0]["retrieval"]["dense_score"] > 0.0


def test_pgvector_upsert_preserves_jsonb_fallback():
    """即便开启 pgvector，embedding_json 仍同步写，保证可读/可回退。"""
    store = PostgresLongTermMemoryStore.from_environment()
    store.initialize()

    memory_id = store.upsert(
        {
            "memory_type": "fact",
            "statement": "JSONB fallback preserved under pgvector marker omega-4.",
            "tags": ["jsonb", "fallback"],
            "user_scope": "pgvector-user",
            "project_scope": "pgvector-project",
            "evidence": _evidence("pgvector-jsonb", 4, 5),
        }
    )

    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT embedding_json, embedding FROM long_term_memories WHERE memory_id = %s",
            (memory_id,),
        )
        embedding_json, embedding = cursor.fetchone()
    assert embedding_json is not None and embedding_json != "[]"
    assert embedding is not None  # 两边都写，缺任何一边都不属“全量回退”
