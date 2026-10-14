"""Persistent local wiki for SciCoreAgent.

The wiki captures factual knowledge points and frequently asked questions
that appear across user sessions.  It is intentionally a flat JSONL file
(like the evidence store) so users can review, edit, or delete entries
without a database.

Format (one JSON object per line in .scicore/wiki/entries.jsonl):
    {"id": "uuid", "title": "...", "content": "...", "tags": [...],
     "source_session": "...", "created_at": "..."}

Auto-trigger mechanism:
    1. Explicit intent: user says "记住"/"记下来"/"remember this" → save
    2. Duplicate detection: same question asked 2+ times → auto-save
    3. Auto-search: before answering, search wiki for relevant past knowledge
    4. Deduplication: skip save if similar entry already exists
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WIKI_DIR = Path(".scicore") / "wiki"
ENTRIES_FILE = "entries.jsonl"
QUESTIONS_FILE = "questions.jsonl"

# Keywords that signal the user explicitly wants something remembered.
_EXPLICIT_REMEMBER_PATTERNS = [
    r"记下[来]?",
    r"记住",
    r"保存下[来]?",
    r"沉淀下[来]?",
    r"记录下[来]?",
    r"帮我记",
    r"remember\s+this",
    r"save\s+this",
    r"note\s+this",
    r"make\s+a?\s*note",
    r"备忘",
    r"存到.*wiki",
    r"存到.*知识库",
]

# Similarity threshold for dedup: fraction of query terms that must match.
SIMILARITY_THRESHOLD = 0.5
# Auto-save when the same question appears this many times.
DUPLICATE_TRIGGER_COUNT = 2


def wiki_dir(root: str | Path | None = None) -> Path:
    """Resolve the wiki data directory."""
    return (Path(root or ".").resolve() / DEFAULT_WIKI_DIR).resolve()


def _ensure_dir(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)


def _read_entries(root: str | Path | None = None) -> list[dict]:
    """Read all wiki entries from entries.jsonl.  Returns empty list if missing."""
    path = wiki_dir(root) / ENTRIES_FILE
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _write_entries(entries: list[dict], root: str | Path | None = None) -> None:
    """Overwrite entries.jsonl with the given list of entries."""
    base = wiki_dir(root)
    _ensure_dir(base)
    path = base / ENTRIES_FILE
    lines = "\n".join(
        json.dumps(e, ensure_ascii=False, default=str) for e in entries
    )
    path.write_text(lines + "\n", encoding="utf-8")


def save_entry(
    title: str,
    content: str,
    tags: list[str] | None = None,
    source_session: str = "",
    root: str | Path | None = None,
) -> dict:
    """Append a knowledge entry to the wiki.

    Returns the saved entry dict.  Raises ValueError if title or content
    is empty.
    """
    title = title.strip()
    content = content.strip()
    if not title:
        raise ValueError("wiki entry title must not be empty")
    if not content:
        raise ValueError("wiki entry content must not be empty")

    clean_tags = _normalize_tags(tags or [])
    entry = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "content": content,
        "tags": clean_tags,
        "source_session": source_session.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    base = wiki_dir(root)
    _ensure_dir(base)
    path = base / ENTRIES_FILE
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def search_entries(
    query: str,
    limit: int = 10,
    root: str | Path | None = None,
) -> list[dict]:
    """Keyword search across wiki entries.

    Uses _tokenize() which supports both Chinese (CJK) and English tokens
    with stop-word filtering.
    """
    terms = _tokenize(query)
    if not terms:
        return _read_entries(root)[:limit]

    scored: list[tuple[int, str, dict]] = []
    for entry in _read_entries(root):
        text = json.dumps(entry, ensure_ascii=False, default=str).lower()
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, entry.get("created_at", ""), entry))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def list_entries(
    limit: int = 20,
    root: str | Path | None = None,
) -> list[dict]:
    """Return the most recent wiki entries (newest first)."""
    entries = _read_entries(root)
    entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return entries[:limit]


def get_entry(
    entry_id: str,
    root: str | Path | None = None,
) -> dict | None:
    """Look up a single wiki entry by its id."""
    for entry in _read_entries(root):
        if entry.get("id") == entry_id:
            return entry
    return None


def wiki_prompt_context(
    root: str | Path | None = None,
    limit: int = 5,
    max_chars: int = 2000,
) -> str:
    """Return a compact prompt section describing recent wiki entries.

    Analogous to skills_prompt_context() in skills.py.
    """
    entries = list_entries(limit=limit, root=root)
    if not entries:
        return "No wiki entries yet."

    lines = []
    for entry in entries:
        tags = entry.get("tags", [])
        tag_text = f" [{', '.join(tags)}]" if tags else ""
        snippet = entry.get("content", "")[:120].replace("\n", " ")
        lines.append(f"- **{entry['title']}**{tag_text}: {snippet}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... wiki list truncated ..."
    return text


# ---------------------------------------------------------------------------
# Auto-trigger: duplicate detection, explicit intent, dedup
# ---------------------------------------------------------------------------


def detect_explicit_remember(user_message: str) -> bool:
    """Return True if the user explicitly asks to remember/save knowledge."""
    text = user_message.lower().strip()
    for pattern in _EXPLICIT_REMEMBER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def find_similar_entries(
    query: str,
    threshold: float = SIMILARITY_THRESHOLD,
    root: str | Path | None = None,
) -> list[dict]:
    """Return existing wiki entries that are similar to *query*.

    Similarity is measured as the fraction of query tokens that appear
    in the entry.  Supports Chinese (CJK) and English tokens.
    Entries with a score >= *threshold* are returned.
    Returns empty list if nothing is similar enough — meaning it's safe
    to save a new entry without creating a near-duplicate.
    """
    terms = _tokenize(query)
    if not terms:
        return []

    results: list[tuple[float, dict]] = []
    for entry in _read_entries(root):
        text = json.dumps(entry, ensure_ascii=False, default=str).lower()
        matched = sum(1 for term in terms if term in text)
        score = matched / len(terms)
        if score >= threshold:
            results.append((score, entry))

    results.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in results]


def is_duplicate_question(
    question: str,
    threshold: float = SIMILARITY_THRESHOLD,
    root: str | Path | None = None,
) -> bool:
    """Check if a similar entry already exists in the wiki.

    Shortcut: if find_similar_entries returns anything, it's a duplicate.
    """
    return len(find_similar_entries(question, threshold, root)) > 0


# ---------------------------------------------------------------------------
# Session-level question tracker
# ---------------------------------------------------------------------------


class QuestionTracker:
    """Tracks user questions within and across sessions.

    Stores a log of questions in .scicore/wiki/questions.jsonl.
    When the same question (or a very similar one) appears
    DUPLICATE_TRIGGER_COUNT times, it fires a signal that the agent
    should auto-save the knowledge to the wiki.
    """

    def __init__(self, root: str | Path | None = None):
        self._base = wiki_dir(root)
        self._ensure()

    def _ensure(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)

    def _questions_path(self) -> Path:
        return self._base / QUESTIONS_FILE

    def _read_questions(self) -> list[dict]:
        path = self._questions_path()
        if not path.exists():
            return []
        questions: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(json.loads(line))
        return questions

    def record(self, question: str, answer_summary: str = "") -> dict:
        """Record a user question. Returns a dict with duplicate info."""
        question = question.strip()
        if not question:
            return {"is_duplicate": False, "count": 0}

        # Check similarity against past questions.
        past = self._read_questions()
        similar: list[dict] = []
        terms = _tokenize(question)

        for past_q in past:
            past_terms = _tokenize(past_q.get("question", ""))
            if terms and past_terms:
                common = terms & past_terms
                score = len(common) / max(len(terms), len(past_terms))
                if score >= SIMILARITY_THRESHOLD:
                    similar.append(past_q)

        count = len(similar) + 1  # +1 for this occurrence

        # Persist this question.
        entry = {
            "question": question,
            "answer_summary": answer_summary[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "occurrence": count,
        }
        path = self._questions_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        return {
            "is_duplicate": count >= DUPLICATE_TRIGGER_COUNT,
            "count": count,
            "similar_questions": [s.get("question", "")[:100] for s in similar[:3]],
        }


def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from text for similarity comparison."""
    # Keep Chinese characters (CJK) and ASCII word characters.
    tokens = re.findall(r"[\w一-鿿]+", text.lower())
    # Filter out very short / stop-word tokens.
    stop = {"the", "a", "an", "is", "are", "was", "were", "do", "does",
            "to", "of", "in", "for", "on", "and", "or", "it", "its",
            "this", "that", "what", "how", "can", "could", "would",
            "的", "了", "是", "我", "你", "他", "她", "它", "们",
            "这", "那", "吗", "呢", "吧", "啊", "哦", "嗯"}
    return {t for t in tokens if len(t) > 1 and t not in stop}


def _normalize_tags(tags: list[str]) -> list[str]:
    """Clean and deduplicate tag strings."""
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = re.sub(r"[^a-z0-9_-]+", "-", tag.strip().lower()).strip("-")
        if normalized and normalized not in seen:
            clean.append(normalized)
            seen.add(normalized)
    return clean
