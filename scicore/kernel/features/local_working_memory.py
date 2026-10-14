"""Small, inspectable User/Project working-memory tier.

This tier is deliberately local and low-volume.  It is independent from the
session JSON, survives a session reset, and mirrors its structured rows to
Markdown so users can inspect what the agent currently treats as reusable
preferences and project decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def _tokens(text):
    text = str(text).lower()
    tokens = set(re.findall(r"[A-Za-z0-9_-]+", text))
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.add(segment)
        tokens.update(segment)
        tokens.update(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _safe_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "local")).strip(".-")
    return text[:72] or "local"


class LocalWorkingMemoryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.db_path = self.root / "working.sqlite"
        self._init_db()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS working_memories (
                    memory_id TEXT PRIMARY KEY,
                    scope_kind TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_working_scope ON working_memories(scope_kind, scope_id, updated_at)"
            )
            connection.commit()

    def upsert(self, memory):
        memory_type = str(memory.get("memory_type", "fact"))
        if memory_type == "user_preference":
            scope_kind = "user"
            scope_id = str(memory.get("user_scope", "") or "local")
        else:
            scope_kind = "project"
            scope_id = str(memory.get("project_scope", "") or "local")
        statement = str(memory.get("statement", "")).strip()
        if not statement:
            return None
        evidence = dict(memory.get("evidence", {}))
        canonical = "\0".join([scope_kind, scope_id, memory_type, statement])
        memory_id = str(memory.get("memory_id") or "work_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24])
        timestamp = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO working_memories (
                    memory_id, scope_kind, scope_id, memory_type, statement,
                    tags_json, source_uri, source_sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    tags_json=excluded.tags_json,
                    source_uri=excluded.source_uri,
                    source_sha256=excluded.source_sha256,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    scope_kind,
                    scope_id,
                    memory_type,
                    statement,
                    json.dumps(memory.get("tags", []), ensure_ascii=False),
                    str(evidence.get("source_uri", "")),
                    str(evidence.get("source_sha256", "")),
                    timestamp,
                ),
            )
            connection.commit()
        self._write_scope_markdown(scope_kind, scope_id)
        return memory_id

    def retrieve(self, query, *, user_scope="", project_scope="", limit=10):
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        clauses = ["scope_kind = 'user' AND (scope_id = 'local' OR scope_id = ?)"]
        values = [str(user_scope or "local")]
        clauses.append("scope_kind = 'project' AND (scope_id = 'local' OR scope_id = ?)")
        values.append(str(project_scope or "local"))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, memory_type, statement, tags_json, source_uri,
                       source_sha256, updated_at
                FROM working_memories WHERE (""" + ") OR (".join(clauses) + ") ORDER BY updated_at DESC",
                tuple(values),
            ).fetchall()
        ranked = []
        for row in rows:
            tags = json.loads(row["tags_json"])
            overlap = len(query_tokens & _tokens(" ".join([row["statement"], *tags])))
            if not overlap:
                continue
            score = overlap / max(1, len(query_tokens))
            ranked.append(
                (score, {
                    "memory_id": row["memory_id"],
                    "text": row["statement"],
                    "tags": tags,
                    "kind": "local_working",
                    "source": row["source_uri"],
                    "source_anchor": "sha256:" + row["source_sha256"],
                    "created_at": row["updated_at"],
                    "retrieval": {"method": "local_working_keyword", "final_score": round(score, 6)},
                })
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked[: max(0, int(limit))]]

    def render(self, *, user_scope="", project_scope="", limit=6):
        rows = self.list_scoped(user_scope=user_scope, project_scope=project_scope, limit=limit)
        if not rows:
            return "- local_user_project_memory: -"
        return "\n".join(["- local_user_project_memory:", *[f"  - {row['text']}" for row in rows]])

    def list_scoped(self, *, user_scope="", project_scope="", limit=6):
        clauses = ["scope_kind = 'user' AND (scope_id = 'local' OR scope_id = ?)"]
        values = [str(user_scope or "local")]
        clauses.append("scope_kind = 'project' AND (scope_id = 'local' OR scope_id = ?)")
        values.append(str(project_scope or "local"))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, memory_type, statement, tags_json, source_uri,
                       source_sha256, updated_at
                FROM working_memories WHERE (""" + ") OR (".join(clauses) + ") ORDER BY updated_at DESC LIMIT ?",
                (*values, max(0, int(limit))),
            ).fetchall()
        return [
            {
                "memory_id": row["memory_id"],
                "text": row["statement"],
                "tags": json.loads(row["tags_json"]),
                "kind": "local_working",
                "source": row["source_uri"],
                "source_anchor": "sha256:" + row["source_sha256"],
                "created_at": row["updated_at"],
            }
            for row in rows
        ]

    def _write_scope_markdown(self, scope_kind, scope_id):
        directory = self.root / scope_kind
        directory.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT memory_type, statement, source_uri, source_sha256, updated_at
                FROM working_memories
                WHERE scope_kind = ? AND scope_id = ? ORDER BY updated_at DESC
                """,
                (scope_kind, scope_id),
            ).fetchall()
        lines = [f"# {scope_kind.title()} Working Memory", "", f"- scope: {scope_id}", "", "## Notes"]
        for row in rows:
            lines.append(f"- [{row['memory_type']}] {row['statement']}")
            lines.append(f"  - source: {row['source_uri']}")
            lines.append(f"  - sha256: {row['source_sha256']}")
            lines.append(f"  - updated_at: {row['updated_at']}")
        (directory / f"{_safe_name(scope_id)}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
