"""Optional PostgreSQL long-term memory backend.

The local SQLite store remains the offline-first default.  This module is
deliberately dependency-light: PostgreSQL is activated only when a DSN is set
and ``psycopg`` is installed.  Every memory keeps a source URI and content
hash, so retrieval never returns an unattributed model summary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from functools import lru_cache


def _tokens(text):
    text = str(text).lower()
    tokens = re.findall(r"[A-Za-z0-9_-]+", text)
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.append(segment)
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _hash_embedding(text, dimensions=256):
    vector = [0.0] * int(dimensions)
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(vector)
        vector[index] += -1.0 if digest[4] & 1 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine(left, right):
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(float(left[index]) * float(right[index]) for index in range(size))


def _bm25(query_tokens, documents, *, k1=1.5, b=0.75):
    """Return exact BM25 scores for a small scoped candidate set."""
    if not documents or not query_tokens:
        return [0.0] * len(documents)
    lengths = [len(tokens) for tokens in documents]
    average = sum(lengths) / max(1, len(lengths))
    scores = [0.0] * len(documents)
    for term in set(query_tokens):
        document_frequency = sum(1 for tokens in documents if term in tokens)
        if not document_frequency:
            continue
        idf = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
        for index, tokens in enumerate(documents):
            frequency = tokens.count(term)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * lengths[index] / max(1.0, average))
            scores[index] += idf * frequency * (k1 + 1) / denominator
    return scores


class EmbeddingProvider:
    """Protocol-like base class for a semantic embedding implementation."""

    name = "dense_hash_fallback"

    def embed(self, text):
        return _hash_embedding(text)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    name = "sentence_transformers"

    def __init__(self, model_name):
        self.model_name = str(model_name)
        self.name = f"sentence_transformers:{self.model_name}"
        self.model = _load_sentence_transformer(self.model_name)

    def embed(self, text):  # pragma: no cover - model is optional in CI
        return [float(value) for value in self.model.encode(str(text), normalize_embeddings=True).tolist()]


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name):
    """Share the heavyweight local model across runtime/pipeline stores."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sentence-transformers is required for BIOCOREAGENT_EMBEDDING_BACKEND=sentence-transformers"
        ) from exc
    return SentenceTransformer(str(model_name))


def embedding_provider_from_environment():
    backend = os.environ.get("BIOCOREAGENT_EMBEDDING_BACKEND", "hash").strip().lower()
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbeddingProvider(
            os.environ.get("BIOCOREAGENT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    return EmbeddingProvider()


class PostgresLongTermMemoryStore:
    """Postgres-backed distilled memory with dense + exact BM25 fusion.

    The implementation deliberately calculates BM25 over the filtered scope in
    Python.  That gives the same BM25 semantics with stock PostgreSQL; a later
    pgvector/ParadeDB deployment can replace only the candidate-fetch method,
    without changing evidence or score semantics.
    """

    def __init__(self, dsn, *, embedding_provider=None):
        self.dsn = str(dsn)
        self.embedding_provider = embedding_provider or embedding_provider_from_environment()
        self._research_graph = None
        # Whether the backing Postgres actually ships the `vector` extension.
        # Detected idempotently in initialize(); when absent the store keeps the
        # JSONB cosine fallback exactly as before.
        self.pgvector_enabled = False

    @classmethod
    def from_environment(cls):
        dsn = os.environ.get("BIOCOREAGENT_MEMORY_POSTGRES_DSN", "").strip()
        return cls(dsn) if dsn else None

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Postgres memory is configured but psycopg is not installed. "
                "Install Biocoreagent's memory-postgres extra."
            ) from exc
        return psycopg.connect(self.dsn)

    def initialize(self):
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_scope TEXT NOT NULL DEFAULT '',
                    project_scope TEXT NOT NULL DEFAULT '',
                    memory_type TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    embedding_json JSONB NOT NULL,
                    embedding_model TEXT NOT NULL,
                    reliability DOUBLE PRECISION NOT NULL DEFAULT 0.65,
                    state TEXT NOT NULL DEFAULT 'active',
                    supersedes TEXT NOT NULL DEFAULT '',
                    distiller_version TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory_evidence (
                    memory_id TEXT NOT NULL REFERENCES long_term_memories(memory_id),
                    source_uri TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_session_id TEXT NOT NULL DEFAULT '',
                    start_message_index INTEGER,
                    end_message_index INTEGER,
                    PRIMARY KEY(memory_id, source_uri)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_scope_state ON long_term_memories(user_scope, project_scope, state)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_evidence_source ON long_term_memory_evidence(source_session_id, start_message_index)"
            )
            # Optional pgvector (dense) upgrade.  Only runs when the image ships
            # the `vector` extension (e.g. pgvector/pgvector:pg15); a stock
            # postgres:15 raises here and the store transparently falls back to
            # the JSONB cosine path.  Every step is idempotent, so repeated
            # initialize() calls never fail.
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    "ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS embedding vector(256)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ltm_embedding_hnsw "
                    "ON long_term_memories USING hnsw (embedding vector_cosine_ops)"
                )
                self.pgvector_enabled = True
            except Exception:
                # Stock Postgres without pgvector (or an in-progress migration):
                # keep the JSONB/RRF behaviour, never wedge startup on a DENY.
                self.pgvector_enabled = False
        return self

    def upsert(self, memory):
        self.initialize()
        statement = str(memory["statement"]).strip()
        if not statement:
            raise ValueError("distilled memory statement is required")
        memory_id = str(memory.get("memory_id") or self._memory_id(memory))
        now = datetime.now(timezone.utc).isoformat()
        embedding = self.embedding_provider.embed(statement)
        evidence = dict(memory["evidence"])
        # JSONB is always written so the store stays readable even when pgvector
        # is not installed; the optional vector column is written on top of it
        # only when the migration succeeded (pgvector_enabled).
        insert_cols = [
            "memory_id", "user_scope", "project_scope", "memory_type", "statement",
            "tags_json", "embedding_json", "embedding_model", "reliability", "state",
            "supersedes", "distiller_version", "created_at", "updated_at",
        ]
        insert_placeholders = [
            "%s", "%s", "%s", "%s", "%s", "%s::jsonb", "%s::jsonb", "%s", "%s",
            "%s", "%s", "%s", "%s", "%s",
        ]
        params = [
            memory_id,
            str(memory.get("user_scope", "")),
            str(memory.get("project_scope", "")),
            str(memory.get("memory_type", "fact")),
            statement,
            json.dumps(memory.get("tags", []), ensure_ascii=False),
            json.dumps(embedding),
            self.embedding_provider.name,
            float(memory.get("reliability", 0.65)),
            str(memory.get("state", "active")),
            str(memory.get("supersedes", "")),
            str(memory.get("distiller_version", "rules_v1")),
            str(memory.get("created_at", now)),
            now,
        ]
        update_set = [
            "statement = EXCLUDED.statement",
            "tags_json = EXCLUDED.tags_json",
            "embedding_json = EXCLUDED.embedding_json",
            "embedding_model = EXCLUDED.embedding_model",
            "reliability = EXCLUDED.reliability",
            "state = EXCLUDED.state",
            "supersedes = EXCLUDED.supersedes",
            "updated_at = EXCLUDED.updated_at",
        ]
        if self.pgvector_enabled:
            insert_cols.append("embedding")
            insert_placeholders.append("%s::vector")
            params.append("[" + ",".join(f"{float(value):.6f}" for value in embedding) + "]")
            update_set.append("embedding = EXCLUDED.embedding")
        sql = (
            "INSERT INTO long_term_memories (" + ", ".join(insert_cols) + ") VALUES ("
            + ", ".join(insert_placeholders) + ") ON CONFLICT(memory_id) DO UPDATE SET "
            + ", ".join(update_set)
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            cursor.execute(
                """
                INSERT INTO long_term_memory_evidence (
                    memory_id, source_uri, source_sha256, source_session_id,
                    start_message_index, end_message_index
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(memory_id, source_uri) DO NOTHING
                """,
                (
                    memory_id,
                    str(evidence["source_uri"]),
                    str(evidence["source_sha256"]),
                    str(evidence.get("session_id", "")),
                    evidence.get("start_message_index"),
                    evidence.get("end_message_index"),
                ),
            )
        return memory_id

    def upsert_graph_memory(self, memory):
        """Project one attributed memory into the temporal research graph."""
        if self._research_graph is None:
            from .research_graph import PostgresResearchGraphStore

            self._research_graph = PostgresResearchGraphStore(self.dsn)
        return self._research_graph.upsert_memory(memory)

    @property
    def research_graph(self):
        if self._research_graph is None:
            from .research_graph import PostgresResearchGraphStore

            self._research_graph = PostgresResearchGraphStore(self.dsn)
        return self._research_graph

    @staticmethod
    def _memory_id(memory):
        canonical = "\0".join(
            [
                str(memory.get("memory_type", "fact")),
                str(memory.get("statement", "")),
                str(memory.get("user_scope", "")),
                str(memory.get("project_scope", "")),
            ]
        )
        return "mem_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def retrieve(self, query, *, limit=10, user_scope="", project_scope=""):
        self.initialize()
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        where = ["memory.state = 'active'"]
        values = []
        if user_scope:
            where.append("(memory.user_scope = '' OR memory.user_scope = %s)")
            values.append(str(user_scope))
        else:
            # An unscoped local process may read only explicitly global rows,
            # never every tenant's private memory.
            where.append("memory.user_scope = ''")
        if project_scope:
            where.append("(memory.project_scope = '' OR memory.project_scope = %s)")
            values.append(str(project_scope))
        else:
            where.append("memory.project_scope = ''")
        dense_query = self.embedding_provider.embed(query)
        # pgvector path: ask the DB for the native cosine distance on the SAME
        # scope-filtered candidate set, then RRF-fuse with BM25 (still computed
        # in Python over the filtered candidates).  When pgvector is not
        # installed the store keeps the JSONB cosine fallback, unchanged.
        if self.pgvector_enabled:
            select_clause = (
                "SELECT memory.memory_id, memory.statement, memory.tags_json, "
                "       memory.embedding_json, memory.embedding_model, "
                "       memory.reliability, memory.memory_type, "
                "       evidence.source_uri, evidence.source_sha256, "
                "       evidence.source_session_id, evidence.start_message_index, "
                "       evidence.end_message_index, "
                "       memory.embedding <=> %s::vector AS dense_dist "
                "FROM long_term_memories AS memory "
                "JOIN long_term_memory_evidence AS evidence ON evidence.memory_id = memory.memory_id "
                "WHERE " + " AND ".join(where)
            )
            bind_params = ("[" + ",".join(f"{float(value):.6f}" for value in dense_query) + "]",) + tuple(values)
        else:
            select_clause = (
                "SELECT memory.memory_id, memory.statement, memory.tags_json, "
                "       memory.embedding_json, memory.embedding_model, "
                "       memory.reliability, memory.memory_type, "
                "       evidence.source_uri, evidence.source_sha256, "
                "       evidence.source_session_id, evidence.start_message_index, "
                "       evidence.end_message_index "
                "FROM long_term_memories AS memory "
                "JOIN long_term_memory_evidence AS evidence ON evidence.memory_id = memory.memory_id "
                "WHERE " + " AND ".join(where)
            )
            bind_params = tuple(values)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(select_clause, bind_params)
            columns = [item.name for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        documents = [_tokens(row["statement"] + " " + " ".join(row["tags_json"] or [])) for row in rows]
        bm25_scores = _bm25(query_tokens, documents)
        dense_scores = []
        for row in rows:
            if row["embedding_model"] != self.embedding_provider.name:
                dense_scores.append(0.0)
            elif self.pgvector_enabled:
                # `<=>` is cosine distance; keep the gate, only reuse the vector
                # column (dense_dist) when it is actually populated.
                distance = row.get("dense_dist")
                dense_scores.append(max(0.0, 1.0 - float(distance)) if distance is not None else 0.0)
            else:
                dense_scores.append(_cosine(dense_query, row["embedding_json"] or []))
        bm25_order = sorted(range(len(rows)), key=lambda index: bm25_scores[index], reverse=True)
        dense_order = sorted(range(len(rows)), key=lambda index: dense_scores[index], reverse=True)
        bm25_ranks = {index: rank + 1 for rank, index in enumerate(bm25_order) if bm25_scores[index] > 0}
        dense_ranks = {index: rank + 1 for rank, index in enumerate(dense_order) if dense_scores[index] > 0}
        ranked = []
        for index, row in enumerate(rows):
            b_rank, d_rank = bm25_ranks.get(index), dense_ranks.get(index)
            if b_rank is None and d_rank is None:
                continue
            rrf = (0.5 / (60 + b_rank) if b_rank else 0.0) + (0.5 / (60 + d_rank) if d_rank else 0.0)
            ranked.append(
                (rrf, {
                    "memory_id": row["memory_id"],
                    "text": row["statement"],
                    "tags": row["tags_json"] or [],
                    "kind": "postgres_distilled",
                    "source": row["source_uri"],
                    "source_anchor": "sha256:" + row["source_sha256"],
                    "source_session_id": row["source_session_id"],
                    "start_message_index": row["start_message_index"],
                    "end_message_index": row["end_message_index"],
                    "retrieval": {
                        "method": "rrf_dense_bm25",
                        "embedding_model": row["embedding_model"],
                        "bm25_rank": b_rank,
                        "dense_rank": d_rank,
                        "bm25_score": round(bm25_scores[index], 6),
                        "dense_score": round(dense_scores[index], 6),
                        "rrf_score": round(rrf, 8),
                    },
                })
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        output = []
        seen_memory_ids = set()
        for _, item in ranked:
            if item["memory_id"] in seen_memory_ids:
                continue
            seen_memory_ids.add(item["memory_id"])
            output.append(item)
            if len(output) >= max(0, int(limit)):
                break
        return output
