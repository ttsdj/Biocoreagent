"""Durable session persistence with SQLite primary storage and JSON mirrors."""

import json
import hashlib
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sessions.sqlite"
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self):
        with self._lock, closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    workspace_root TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)"
            )
            # The JSON payload above remains the backwards-compatible session
            # snapshot.  These normalized tables are the audit/source-of-truth
            # index used by cross-session memory and source backlinks.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    tool_name TEXT NOT NULL DEFAULT '',
                    args_json TEXT NOT NULL DEFAULT '{}',
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (session_id, message_index)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages(session_id, message_index)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    start_message_index INTEGER NOT NULL,
                    end_message_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, start_message_index, end_message_index, content_sha256)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_chunks_session ON conversation_chunks(session_id, start_message_index)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS distillation_outbox (
                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_distillation_outbox_status ON distillation_outbox(status, outbox_id)"
            )
            connection.commit()

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        session_id = str(session["id"])
        payload = json.dumps(session, ensure_ascii=False, separators=(",", ":"))
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, workspace_root, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    workspace_root=excluded.workspace_root,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (
                    session_id,
                    str(session.get("workspace_root", "")),
                    str(session.get("created_at", "")),
                    updated_at,
                    payload,
                ),
            )
            connection.commit()

            self._sync_messages(connection, session_id, session.get("history", []))

            # Commit normalized evidence in the same save cycle as the session
            # snapshot.  We intentionally never delete historical messages.
            connection.commit()

        path = self.path(session_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @staticmethod
    def _content_hash(value):
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def _sync_messages(self, connection, session_id, history):
        for index, item in enumerate(history or []):
            content = str(item.get("content", ""))
            args = item.get("args", {}) if isinstance(item, dict) else {}
            connection.execute(
                """
                INSERT INTO session_messages (
                    session_id, message_index, role, tool_name, args_json,
                    content, content_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, message_index) DO UPDATE SET
                    role=excluded.role,
                    tool_name=excluded.tool_name,
                    args_json=excluded.args_json,
                    content=excluded.content,
                    content_sha256=excluded.content_sha256,
                    created_at=excluded.created_at
                """,
                (
                    session_id,
                    index,
                    str(item.get("role", "")),
                    str(item.get("name", "")),
                    json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    content,
                    self._content_hash(content),
                    str(item.get("created_at", "")),
                ),
            )

    def create_chunks(self, session_id, *, max_chars=1800, overlap_chars=180, strategy="fixed"):
        """Chunk a persisted session and return immutable source-addressable rows.

        Chunk ids include the exact content hash, so a later edited/rewritten
        history never silently changes the evidence behind an existing memory.

        Args:
            strategy:
                "fixed"    — 默认，按字符上限在消息边界处贪心打包（保持原有行为）。
                "semantic" — 在角色/工具切换、段落空行、标题行（## 等）这类自然
                    边界处优先断块，避免把跨段落/跨语轮的上下文从中间切开。
                    semantic 不再做字符重叠（overlap 仅对 fixed 生效），
                    chunk_id / source_uri（含 #sha256）寻址格式不变。
        """
        session_id = str(session_id)
        max_chars = max(200, int(max_chars))
        overlap_chars = max(0, min(int(overlap_chars), max_chars // 2))
        if strategy == "semantic":
            overlap_chars = 0
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT message_index, role, tool_name, args_json, content, created_at
                FROM session_messages WHERE session_id = ? ORDER BY message_index
                """,
                (session_id,),
            ).fetchall()
            blocks = []
            for row in rows:
                label = str(row["role"])
                if row["tool_name"]:
                    label += ":" + str(row["tool_name"])
                blocks.append(
                    {
                        "index": int(row["message_index"]),
                        "label": label,
                        "text": f"[{label}] {row['content']}",
                        "content": str(row["content"]),
                        "created_at": str(row["created_at"]),
                    }
                )
            if strategy == "semantic":
                return self._pack_semantic(connection, session_id, blocks, max_chars)
            created = []
            start = 0
            while start < len(blocks):
                selected = []
                size = 0
                cursor = start
                while cursor < len(blocks):
                    block = blocks[cursor]
                    addition = len(block["text"]) + (1 if selected else 0)
                    if selected and size + addition > max_chars:
                        break
                    selected.append(block)
                    size += addition
                    cursor += 1
                if not selected:
                    selected = [blocks[start]]
                    cursor = start + 1
                content = "\n".join(item["text"] for item in selected)
                digest = self._content_hash(content)
                first, last = selected[0]["index"], selected[-1]["index"]
                chunk_id = f"chunk_{session_id}_{first}_{last}_{digest[:16]}"
                source_uri = f"session://{session_id}/messages/{first}-{last}#sha256={digest}"
                created_at = selected[-1]["created_at"] or datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT OR IGNORE INTO conversation_chunks (
                        chunk_id, session_id, start_message_index, end_message_index,
                        content, content_sha256, source_uri, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, session_id, first, last, content, digest, source_uri, created_at),
                )
                row = {
                    "chunk_id": chunk_id,
                    "session_id": session_id,
                    "start_message_index": first,
                    "end_message_index": last,
                    "content": content,
                    "content_sha256": digest,
                    "source_uri": source_uri,
                    "created_at": created_at,
                }
                created.append(row)
                if cursor >= len(blocks):
                    break
                if overlap_chars:
                    carried = 0
                    next_start = cursor
                    for reverse_index in range(cursor - 1, start - 1, -1):
                        carried += len(blocks[reverse_index]["text"]) + 1
                        if carried > overlap_chars:
                            break
                        next_start = reverse_index
                    start = max(start + 1, next_start)
                else:
                    start = cursor
            connection.commit()
        return created

    # ------------------------------------------------------------------
    # Semantic chunking (strategy="semantic")
    # ------------------------------------------------------------------

    @staticmethod
    def _is_section_opener(blocks, index):
        """blocks[index] 是否开始一个新的语义段落。

        三个判据（满足任一即为断块点）：
          1. 与前一条消息的角色/工具标签不同（跨语轮/切换工具）；
          2. 本条内容以 Markdown 标题（# / ## 等）开头；
          3. 本条内容内出现段落空行（\\n\\n），视为新段落。
        """
        if index == 0:
            return True
        prev = blocks[index - 1]
        curr = blocks[index]
        if prev["label"] != curr["label"]:
            return True
        if curr["content"].lstrip().startswith("#"):
            return True
        if "\n\n" in curr["content"]:
            return True
        return False

    def _pack_semantic(self, connection, session_id, blocks, max_chars):
        """按语义边界打包：每个 chunk 从一条「新段落」开始，避免从半句切开。"""
        created = []
        index = 0
        while index < len(blocks):
            selected = [blocks[index]]
            size = len(blocks[index]["text"])
            cursor = index + 1
            while cursor < len(blocks):
                addition = len(blocks[cursor]["text"]) + 1
                if self._is_section_opener(blocks, cursor) or size + addition > max_chars:
                    break
                selected.append(blocks[cursor])
                size += addition
                cursor += 1
            created.append(
                self._emit_chunk(connection, session_id, selected)
            )
            index = cursor
        connection.commit()
        return created

    def _emit_chunk(self, connection, session_id, selected):
        """把一段选中的 blocks 落库为一行 conversation_chunk，返回可寻址行。"""
        content = "\n".join(item["text"] for item in selected)
        digest = self._content_hash(content)
        first, last = selected[0]["index"], selected[-1]["index"]
        chunk_id = f"chunk_{session_id}_{first}_{last}_{digest[:16]}"
        source_uri = f"session://{session_id}/messages/{first}-{last}#sha256={digest}"
        created_at = selected[-1]["created_at"] or datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT OR IGNORE INTO conversation_chunks (
                chunk_id, session_id, start_message_index, end_message_index,
                content, content_sha256, source_uri, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, session_id, first, last, content, digest, source_uri, created_at),
        )
        return {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "start_message_index": first,
            "end_message_index": last,
            "content": content,
            "content_sha256": digest,
            "source_uri": source_uri,
            "created_at": created_at,
        }

    def enqueue_distillation(self, chunks, *, metadata=None):
        timestamp = datetime.now(timezone.utc).isoformat()
        queued = 0
        with self._lock, closing(self._connect()) as connection:
            for chunk in chunks:
                payload = {"chunk": dict(chunk), "metadata": dict(metadata or {})}
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO distillation_outbox (
                        chunk_id, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (str(chunk["chunk_id"]), json.dumps(payload, ensure_ascii=False), timestamp, timestamp),
                )
                queued += int(cursor.rowcount > 0)
            connection.commit()
        return queued

    def pending_distillations(self, limit=50):
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT outbox_id, chunk_id, payload_json, attempts
                FROM distillation_outbox WHERE status = 'pending'
                ORDER BY outbox_id LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "outbox_id": int(row["outbox_id"]),
                "chunk_id": str(row["chunk_id"]),
                "payload": json.loads(row["payload_json"]),
                "attempts": int(row["attempts"]),
            }
            for row in rows
        ]

    def mark_distillation(self, outbox_id, *, status, error=""):
        if status not in {"pending", "processed", "rejected", "failed"}:
            raise ValueError(f"invalid distillation status: {status}")
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE distillation_outbox
                SET status = ?, attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE outbox_id = ?
                """,
                (status, str(error), datetime.now(timezone.utc).isoformat(), int(outbox_id)),
            )
            connection.commit()

    def source_excerpt(self, session_id, start_message_index, end_message_index):
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT message_index, role, tool_name, args_json, content, content_sha256, created_at
                FROM session_messages
                WHERE session_id = ? AND message_index BETWEEN ? AND ?
                ORDER BY message_index
                """,
                (str(session_id), int(start_message_index), int(end_message_index)),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, *, session_id=None, limit=1000):
        """Return persisted chunks for benchmark curation without raw DB access."""
        where = ""
        values = []
        if session_id:
            where = "WHERE session_id = ?"
            values.append(str(session_id))
        values.append(max(1, int(limit)))
        with self._lock, closing(self._connect()) as connection:
            sql = (
                "SELECT chunk_id, session_id, start_message_index, end_message_index, "
                "content, content_sha256, source_uri, created_at "
                "FROM conversation_chunks " + where + " ORDER BY created_at, chunk_id LIMIT ?"
            )
            rows = connection.execute(
                sql,
                tuple(values),
            ).fetchall()
        return [dict(row) for row in rows]

    def session_ids(self, *, limit=1000):
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        if rows:
            return [str(row["session_id"]) for row in rows]
        return [path.stem for path in sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[: max(1, int(limit))]]

    def load(self, session_id):
        session_id = str(session_id)
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is not None:
            return json.loads(row["payload_json"])

        path = self.path(session_id)
        session = json.loads(path.read_text(encoding="utf-8"))
        self.save(session)
        return session

    def latest(self):
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if row is not None:
            return str(row["session_id"])

        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        if not files:
            return None
        session_id = files[-1].stem
        self.load(session_id)
        return session_id
