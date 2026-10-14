"""多步 agent 运行时使用的轻量工作记忆。

session history 负责保存完整事件流；这个模块只保存更小的一层工作集：
当前任务摘要、最近接触的文件、文件短摘要，以及少量跨轮笔记。
这样下一轮 prompt 还能接上上一轮，但不会被整段历史塞满。
"""

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime
import re
from pathlib import Path

from ..workspace import clip, now
from .local_working_memory import LocalWorkingMemoryStore
from .postgres_memory import PostgresLongTermMemoryStore

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
DEFAULT_RETRIEVAL_WEIGHTS = {
    "relevance": 0.45,
    "recency": 0.15,
    "reliability": 0.25,
    "scope_fit": 0.15,
}
PROTECTED_MEMORY_TOPICS = {"user-preferences", "key-decisions"}

DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}


def default_memory_state():
    # 用一个小而结构化的状态，而不是一大段自由文本摘要。
    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


class DurableMemoryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.index_path = self.root / "MEMORY.md"
        self.topics_dir = self.root / "topics"
        self.db_path = self.root / "memory.sqlite"
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
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_anchor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reliability REAL NOT NULL DEFAULT 0.65,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'active',
                    superseded_by TEXT NOT NULL DEFAULT '',
                    UNIQUE(topic, text)
                )
                """
            )
            existing_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(memories)").fetchall()
            }
            migrations = {
                "reliability": "REAL NOT NULL DEFAULT 0.65",
                "access_count": "INTEGER NOT NULL DEFAULT 0",
                "success_count": "INTEGER NOT NULL DEFAULT 0",
                "last_accessed_at": "TEXT NOT NULL DEFAULT ''",
                "state": "TEXT NOT NULL DEFAULT 'active'",
                "superseded_by": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    connection.execute(f"ALTER TABLE memories ADD COLUMN {column} {definition}")
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    search_text,
                    content='memories',
                    content_rowid='id'
                )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, search_text)
                    VALUES (new.id, new.search_text);
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, search_text)
                    VALUES ('delete', old.id, old.search_text);
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, search_text)
                    VALUES ('delete', old.id, old.search_text);
                    INSERT INTO memories_fts(rowid, search_text)
                    VALUES (new.id, new.search_text);
                END
                """
            )
            connection.commit()

    def topic_slugs(self):
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self):
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None
        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def load_topic_notes(self, topic):
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        notes = []
        capture = False
        updated_at = ""
        tags = []
        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now(),
                        "kind": "durable",
                    }
                )
        return notes

    @staticmethod
    def _subject_key(text):
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(_tokenize(match.group(1)))
                return subject or None
        return None

    def retrieval_candidates(
        self,
        query,
        limit=3,
        *,
        scope="",
        role="",
        weights=None,
        half_life_days=90.0,
    ):
        self._sync_markdown_topics()
        query_tokens = _feature_tokens(query)
        if not query_tokens:
            return []

        retrieval_weights = _normalized_weights(weights)
        scope_tokens = _feature_tokens(" ".join([str(scope), str(role)]))
        query_vector = _hash_vector(query_tokens)
        bm25_ranks = {}
        match_query = " OR ".join(f'"{token}"' for token in sorted(query_tokens))
        with closing(self._connect()) as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT memories.id, bm25(memories_fts) AS rank
                    FROM memories_fts
                    JOIN memories ON memories.id = memories_fts.rowid
                    WHERE memories_fts MATCH ? AND memories.state = 'active'
                    ORDER BY rank
                    LIMIT 100
                    """,
                    (match_query,),
                ).fetchall()
                bm25_ranks = {int(row["id"]): index + 1 for index, row in enumerate(rows)}
            except sqlite3.OperationalError:
                bm25_ranks = {}

            rows = connection.execute(
                """
                SELECT id, topic, text, tags_json, vector_json,
                       source_path, source_anchor, created_at,
                       reliability, access_count, success_count,
                       last_accessed_at, state, superseded_by
                FROM memories
                WHERE state = 'active'
                """
            ).fetchall()

        dense_rows = []
        for row in rows:
            dense_score = _cosine_similarity(
                query_vector,
                {int(key): float(value) for key, value in json.loads(row["vector_json"]).items()},
            )
            if dense_score > 0:
                dense_rows.append((int(row["id"]), dense_score))
        dense_rows.sort(key=lambda item: item[1], reverse=True)
        dense_ranks = {row_id: index + 1 for index, (row_id, _) in enumerate(dense_rows)}
        dense_scores = dict(dense_rows)

        ranked = []
        for row in rows:
            row_id = int(row["id"])
            bm25_rank = bm25_ranks.get(row_id)
            dense_rank = dense_ranks.get(row_id)
            if bm25_rank is None and dense_rank is None:
                continue
            reciprocal_rank = 0.0
            if bm25_rank is not None:
                reciprocal_rank += 0.55 / (60 + bm25_rank)
            if dense_rank is not None:
                reciprocal_rank += 0.45 / (60 + dense_rank)
            lexical_relevance = 0.0 if bm25_rank is None else 1.0 / (1.0 + math.log1p(bm25_rank))
            relevance = max(
                0.0,
                min(
                    1.0,
                    0.55 * lexical_relevance
                    + 0.45 * max(0.0, dense_scores.get(row_id, 0.0)),
                ),
            )
            age_days = max(
                0.0,
                (datetime.now().timestamp() - _parse_timestamp(row["created_at"])) / 86400.0,
            )
            recency = math.exp(-math.log(2) * age_days / max(1.0, float(half_life_days)))
            reliability = max(0.0, min(1.0, float(row["reliability"])))
            memory_scope_tokens = _feature_tokens(
                " ".join(
                    [
                        str(row["topic"]),
                        str(row["source_path"]),
                        *json.loads(row["tags_json"]),
                    ]
                )
            )
            if not scope_tokens:
                scope_fit = 0.5
            else:
                overlap = len(scope_tokens & memory_scope_tokens)
                scope_fit = min(1.0, 0.25 + 0.25 * overlap) if overlap else 0.25
            final_score = (
                relevance * retrieval_weights["relevance"]
                + recency * retrieval_weights["recency"]
                + reliability * retrieval_weights["reliability"]
                + scope_fit * retrieval_weights["scope_fit"]
            )
            ranked.append(
                (
                    final_score,
                    {
                        "memory_id": row_id,
                        "text": row["text"],
                        "tags": json.loads(row["tags_json"]),
                        "source": row["source_path"],
                        "source_anchor": row["source_anchor"],
                        "created_at": row["created_at"],
                        "kind": "durable",
                        "retrieval": {
                            "method": "rrf_bm25_dense_hash",
                            "bm25_rank": bm25_rank,
                            "dense_rank": dense_rank,
                            "dense_score": round(dense_scores.get(row_id, 0.0), 6),
                            "rrf_score": round(reciprocal_rank, 8),
                            "scoring_model": "four_dimension_v1",
                            "relevance": round(relevance, 6),
                            "recency": round(recency, 6),
                            "reliability": round(reliability, 6),
                            "scope_fit": round(scope_fit, 6),
                            "weights": retrieval_weights,
                            "final_score": round(final_score, 6),
                            "age_days": round(age_days, 3),
                            "access_count": int(row["access_count"]),
                            "success_count": int(row["success_count"]),
                        },
                    },
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [note for _, note in ranked[: max(0, int(limit))]]
        if selected:
            accessed_at = now()
            selected_ids = [int(note["memory_id"]) for note in selected]
            placeholders = ",".join("?" for _ in selected_ids)
            with closing(self._connect()) as connection:
                connection.execute(
                    f"""
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    (accessed_at, *selected_ids),
                )
                connection.commit()
        return selected

    def record_feedback(self, memory_id, *, verified, success):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT reliability FROM memories WHERE id = ?",
                (int(memory_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown memory id: {memory_id}")
            current = max(0.0, min(1.0, float(row["reliability"])))
            if verified and success:
                updated = current + 0.12 * (1.0 - current)
            elif verified:
                updated = current * 0.75
            else:
                updated = current
            connection.execute(
                """
                UPDATE memories
                SET reliability = ?,
                    success_count = success_count + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (updated, int(bool(verified and success)), now(), int(memory_id)),
            )
            connection.commit()
        return round(updated, 6)

    def apply_decay(self, *, threshold=0.18, half_life_days=180.0, at_timestamp=None):
        reference = datetime.now().timestamp() if at_timestamp is None else float(at_timestamp)
        forgotten = []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, topic, text, reliability, access_count,
                       created_at, last_accessed_at
                FROM memories
                WHERE state = 'active'
                """
            ).fetchall()
            for row in rows:
                if row["topic"] in PROTECTED_MEMORY_TOPICS:
                    continue
                anchor_time = _parse_timestamp(row["last_accessed_at"]) or _parse_timestamp(
                    row["created_at"]
                )
                age_days = max(0.0, (reference - anchor_time) / 86400.0)
                temporal_value = math.exp(
                    -math.log(2) * age_days / max(1.0, float(half_life_days))
                )
                access_bonus = min(0.25, math.log1p(int(row["access_count"])) / 12.0)
                retained_value = float(row["reliability"]) * temporal_value + access_bonus
                if retained_value >= float(threshold):
                    continue
                connection.execute(
                    "UPDATE memories SET state = 'forgotten', updated_at = ? WHERE id = ?",
                    (now(), int(row["id"])),
                )
                forgotten.append(
                    {
                        "memory_id": int(row["id"]),
                        "topic": row["topic"],
                        "text": row["text"],
                        "retained_value": round(retained_value, 6),
                    }
                )
            connection.commit()
        return forgotten

    def _sync_markdown_topics(self):
        timestamp = now()
        with closing(self._connect()) as connection:
            for topic in self.load_index():
                topic_slug = topic["topic"]
                tags = sorted(set(topic.get("tags", [])))
                source_path = str((self.topics_dir / f"{topic_slug}.md").resolve())
                for note in self.load_topic_notes(topic_slug):
                    text = str(note["text"]).strip()
                    if not text:
                        continue
                    note_tags = sorted(set(tags) | set(note.get("tags", [])))
                    tokens = _feature_tokens(
                        " ".join([topic.get("title", ""), text, *note_tags])
                    )
                    anchor = "sha256:" + hashlib.sha256(
                        f"{topic_slug}\0{text}".encode("utf-8")
                    ).hexdigest()
                    connection.execute(
                        """
                        INSERT INTO memories (
                            topic, text, tags_json, search_text, vector_json,
                            source_path, source_anchor, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(topic, text) DO UPDATE SET
                            tags_json=excluded.tags_json,
                            search_text=excluded.search_text,
                            vector_json=excluded.vector_json,
                            source_path=excluded.source_path,
                            source_anchor=excluded.source_anchor,
                            updated_at=excluded.updated_at
                        """,
                        (
                            topic_slug,
                            text,
                            json.dumps(note_tags, ensure_ascii=False),
                            " ".join(sorted(tokens)),
                            json.dumps(_hash_vector(tokens), sort_keys=True),
                            source_path,
                            anchor,
                            str(note.get("created_at", "") or timestamp),
                            timestamp,
                        ),
                    )
            connection.commit()

    def _write_index(self, topics):
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Durable Memory Index", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_topic(self, topic, notes):
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        (self.topics_dir / f"{topic}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def promote(self, promotions):
        if not promotions:
            return [], []
        topics = {topic["topic"]: topic for topic in self.load_index()}
        topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
        results = []
        superseded = []
        superseded_pairs = []
        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS[topic]
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "tags": list(meta["tags"]),
                },
            )
            existing = topic_notes.setdefault(topic, [])
            if note_text in existing:
                continue
            new_subject = self._subject_key(note_text)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    if self._subject_key(old_text) == new_subject:
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        superseded_pairs.append((topic, old_text, note_text))
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")
        self._write_index([topics[slug] for slug in sorted(topics)])
        for topic, notes in topic_notes.items():
            self._write_topic(topic, notes)
        self._sync_markdown_topics()
        if superseded_pairs:
            with closing(self._connect()) as connection:
                for topic, old_text, new_text in superseded_pairs:
                    connection.execute(
                        """
                        UPDATE memories
                        SET state = 'superseded', superseded_by = ?, updated_at = ?
                        WHERE topic = ? AND text = ?
                        """,
                        (new_text, now(), topic, old_text),
                    )
                connection.commit()
        return results, superseded


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _tokenize(text):
    return _feature_tokens(text)


def _feature_tokens(text):
    normalized = str(text).lower()
    tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", normalized)
    for run in cjk_runs:
        tokens.update(run)
        tokens.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return {token for token in tokens if token}


def _hash_vector(tokens, dimensions=256):
    vector = {}
    for token in sorted(set(tokens)):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] = vector.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0:
        return {}
    return {index: value / norm for index, value in vector.items()}


def _cosine_similarity(left, right):
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _parse_timestamp(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _normalized_weights(weights=None):
    merged = dict(DEFAULT_RETRIEVAL_WEIGHTS)
    if weights:
        unknown = sorted(set(weights) - set(merged))
        if unknown:
            raise ValueError(f"unknown retrieval weights: {', '.join(unknown)}")
        for key, value in weights.items():
            merged[key] = max(0.0, float(value))
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("retrieval weights must contain at least one positive value")
    return {key: round(value / total, 6) for key, value in merged.items()}


def _normalize_note(note, index):
    if isinstance(note, str):
        text = clip(note.strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    if not isinstance(note, dict):
        text = clip(str(note).strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    text = clip(str(note.get("text", "")).strip(), 500)
    tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    source = str(note.get("source", "")).strip()
    created_at = str(note.get("created_at", "")).strip() or now()
    note_index = int(note.get("note_index", index))
    kind = str(note.get("kind", "episodic")).strip() or "episodic"
    return {
        "text": text,
        "tags": _dedupe_preserve_order(tags),
        "source": source,
        "created_at": created_at,
        "note_index": note_index,
        "kind": kind,
    }


def normalize_memory_state(state, workspace_root=None):
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    # 规范化层的作用，是把“磁盘里可能长得不太一样的旧状态”
    # 统一整理成当前 runtime 可直接使用的紧凑结构。
    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), 300)
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []

    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        normalized_notes = []
        for index, note in enumerate(episodic_notes):
            if isinstance(note, str) and not str(note).strip():
                continue
            normalized_notes.append(_normalize_note(note, index))
        episodic_notes = normalized_notes
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), 500)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), 500)
            created_at = now()
            freshness = None
        if not path or not text:
            continue
        normalized_file_summaries[path] = {
            "summary": text,
            "created_at": created_at,
            "freshness": freshness,
        }
    state["file_summaries"] = normalized_file_summaries

    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max([note["note_index"] for note in episodic_notes], default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    durable_root = Path(workspace_root) / ".scicore/kernel" / "memory" if workspace_root is not None else None
    durable_store = DurableMemoryStore(durable_root) if durable_root is not None else None
    state["durable_topics"] = durable_store.topic_slugs() if durable_store is not None else []
    return state


def set_task_summary(state, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    files = [item for item in state["working"]["recent_files"] if item != path]
    files.append(path)
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state

    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] = note["note_index"] + 1

    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state
def set_file_summary(state, path, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    state["file_summaries"].pop(path, None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result, limit=180):
    # 我们不会把完整文件内容塞进记忆层，
    # 这里只保留足够提醒下一轮“刚刚读到了什么”的短摘要。
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    if lines[0].startswith("# "):
        lines = lines[1:]
    if not lines:
        return "(empty)"
    summary = " | ".join(lines[:3])
    return clip(summary, limit)


def retrieval_candidates(
    state,
    query,
    limit=3,
    workspace_root=None,
    *,
    scope="",
    role="",
    weights=None,
    half_life_days=90.0,
):
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query)
    retrieval_weights = _normalized_weights(weights)
    scope_tokens = _feature_tokens(" ".join([str(scope), str(role)]))
    ranked = []
    for note in state["episodic_notes"]:
        # 召回逻辑故意保持简单透明：先看 tag 精确命中，
        # 再看关键词重叠，最后看新旧程度。这里不引入 embedding。
        note_tags = {tag.lower() for tag in note.get("tags", [])}
        note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
        overlap = len(query_tokens & note_tokens)
        if overlap == 0:
            continue
        relevance = min(1.0, overlap / max(1, len(query_tokens)))
        age_days = max(
            0.0,
            (datetime.now().timestamp() - _parse_timestamp(note.get("created_at"))) / 86400.0,
        )
        recency = math.exp(-math.log(2) * age_days / max(1.0, float(half_life_days)))
        reliability = 0.55
        scope_fit = 0.5 if not scope_tokens else (1.0 if scope_tokens & note_tokens else 0.25)
        final_score = (
            relevance * retrieval_weights["relevance"]
            + recency * retrieval_weights["recency"]
            + reliability * retrieval_weights["reliability"]
            + scope_fit * retrieval_weights["scope_fit"]
        )
        rendered = dict(note)
        rendered["retrieval"] = {
            "method": "episodic_four_dimension",
            "scoring_model": "four_dimension_v1",
            "relevance": round(relevance, 6),
            "recency": round(recency, 6),
            "reliability": reliability,
            "scope_fit": round(scope_fit, 6),
            "weights": retrieval_weights,
            "final_score": round(final_score, 6),
            "age_days": round(age_days, 3),
        }
        ranked.append((final_score, rendered))

    if workspace_root is not None:
        durable_store = DurableMemoryStore(Path(workspace_root) / ".scicore/kernel" / "memory")
        for note in durable_store.retrieval_candidates(
            query,
            limit=max(limit * 3, limit),
            scope=scope,
            role=role,
            weights=retrieval_weights,
            half_life_days=half_life_days,
        ):
            ranked.append((float(note["retrieval"]["final_score"]), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state, query, limit=3, workspace_root=None, **kwargs):
    candidates = retrieval_candidates(
        state,
        query,
        limit=limit,
        workspace_root=workspace_root,
        **kwargs,
    )
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def render_memory_text(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    # 这里渲染的是给模型看的紧凑“仪表盘”，不是完整回放。
    # 笔记正文默认不展开，只有在相关召回时才按需拿出来。
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]

    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary", "") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    durable_topics = state.get("durable_topics", [])
    lines.append(f"- durable_topics: {', '.join(durable_topics) or '-'}")
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["file_summaries"]
    )


class LayeredMemory:
    def __init__(self, state=None, workspace_root=None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.durable_store = DurableMemoryStore(Path(workspace_root) / ".scicore/kernel" / "memory") if workspace_root is not None else None
        self.local_working_store = LocalWorkingMemoryStore(Path(workspace_root) / ".scicore" / "memory") if workspace_root is not None else None
        # PostgreSQL is an optional cross-session, cross-machine tier.  The
        # existing local SQLite/Markdown tier remains available without any
        # service configuration.
        self.postgres_store = PostgresLongTermMemoryStore.from_environment()

    def to_dict(self):
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query, limit=3, **kwargs):
        local = retrieval_candidates(
            self.state,
            query,
            limit=max(int(limit), int(limit) * 2),
            workspace_root=self.workspace_root,
            **kwargs,
        )
        user_scope = os.environ.get("SCICORE_MEMORY_USER_SCOPE", "").strip()
        project_scope = str(kwargs.get("scope", "") or self.workspace_root or "")
        working = []
        if self.local_working_store is not None:
            working = self.local_working_store.retrieve(
                query,
                limit=max(int(limit), int(limit) * 2),
                user_scope=user_scope,
                project_scope=project_scope,
            )
        if self.postgres_store is None:
            return self._merge_candidates(working, local, limit=limit)
        try:
            remote = self.postgres_store.retrieve(
                query,
                limit=max(int(limit), int(limit) * 2),
                user_scope=user_scope,
                project_scope=project_scope,
            )
        except Exception:
            # A disconnected optional Postgres tier must not make a local CLI
            # unusable; its durable outbox retains evidence for a later retry.
            return self._merge_candidates(working, local, limit=limit)
        return self._merge_candidates(working, remote, local, limit=limit)

    @staticmethod
    def _merge_candidates(*groups, limit):
        merged = []
        seen = set()
        for group in groups:
            for note in group:
                key = (str(note.get("text", "")), str(note.get("source", "")))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(note)
        return merged[:limit]

    def retrieval_view(self, query, limit=3, **kwargs):
        candidates = self.retrieval_candidates(query, limit=limit, **kwargs)
        lines = ["Relevant memory:"]
        lines.extend(f"- {note['text']}" for note in candidates) if candidates else lines.append("- none")
        return "\n".join(lines)

    def render_memory_text(self):
        rendered = render_memory_text(self.state, self.workspace_root)
        if self.local_working_store is None:
            return rendered
        return "\n".join(
            [
                rendered,
                self.local_working_store.render(
                    user_scope=os.environ.get("SCICORE_MEMORY_USER_SCOPE", "").strip(),
                    project_scope=str(self.workspace_root or ""),
                ),
            ]
        )

    def promote_durable(self, promotions):
        if self.durable_store is None:
            return [], []
        self.state = normalize_memory_state(self.state, self.workspace_root)
        promoted, superseded = self.durable_store.promote(promotions)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded

    def record_retrieval_feedback(self, memory_id, *, verified, success):
        if self.durable_store is None:
            raise RuntimeError("durable memory is not configured")
        return self.durable_store.record_feedback(
            memory_id,
            verified=verified,
            success=success,
        )

    def apply_forgetting(self, **kwargs):
        if self.durable_store is None:
            return []
        return self.durable_store.apply_decay(**kwargs)
