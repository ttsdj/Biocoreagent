"""SQLite + FTS5 RAG store with JSONL backup.

Primary storage: SQLite with FTS5 full-text search
Backup: JSONL append-only files

Supports two record types:
  - Literature articles & review rows
  - Protocol records

Reference: SciCoreAgent 需求文档, Section 6.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .literature_schemas import LiteratureArticle, LiteratureReviewRow
from .protocol_schemas import BioProtocolRecord

logger = logging.getLogger(__name__)

DEFAULT_RAG_DIR = Path(".scicore") / "rag"

LITERATURE_DB = "literature.sqlite"
PROTOCOLS_DB = "protocols.sqlite"
LITERATURE_JSONL = "literature.jsonl"
PROTOCOLS_JSONL = "protocols.jsonl"


class SQLiteRagStore:
    """SQLite-based RAG store with FTS5 full-text search and JSONL backup.

    Usage:
        store = SQLiteRagStore()
        store.init_db()
        store.add_literature_article(article)
        results = store.search_literature("RNA-seq workflow", top_k=10)
        stats = store.stats()
    """

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root or ".").resolve()
        self._rag_dir = self._root / DEFAULT_RAG_DIR
        self._rag_dir.mkdir(parents=True, exist_ok=True)

        self._lit_db_path = self._rag_dir / LITERATURE_DB
        self._prot_db_path = self._rag_dir / PROTOCOLS_DB
        self._lit_jsonl_path = self._rag_dir / LITERATURE_JSONL
        self._prot_jsonl_path = self._rag_dir / PROTOCOLS_JSONL

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables and FTS5 indexes if they don't exist."""
        self._init_literature_db()
        self._init_protocols_db()

    def _init_literature_db(self) -> None:
        conn = sqlite3.connect(str(self._lit_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            # Articles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pmid TEXT UNIQUE NOT NULL,
                    doi TEXT,
                    title TEXT,
                    journal TEXT,
                    year INTEGER,
                    abstract TEXT,
                    authors TEXT,
                    last_author TEXT,
                    corresponding_author TEXT,
                    mesh_terms TEXT,
                    publication_types TEXT,
                    fetched_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            # Review rows table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pmid TEXT NOT NULL,
                    method_name TEXT,
                    software_name TEXT,
                    parameters TEXT,
                    lab_info TEXT,
                    paper_title TEXT,
                    journal TEXT,
                    doi TEXT,
                    year INTEGER,
                    evidence_text TEXT,
                    missing_fields TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            # FTS5 virtual table for full-text search on articles
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                    pmid, title, journal, abstract, authors,
                    mesh_terms, content='articles',
                    content_rowid='id'
                )
            """)
            # FTS5 virtual table for full-text search on review rows
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS review_rows_fts USING fts5(
                    method_name, software_name, parameters,
                    evidence_text, paper_title,
                    content='review_rows',
                    content_rowid='id'
                )
            """)
            # Triggers to keep FTS in sync
            self._create_fts_triggers(conn, "articles", "articles_fts",
                                       ["pmid", "title", "journal", "abstract", "authors", "mesh_terms"])
            self._create_fts_triggers(conn, "review_rows", "review_rows_fts",
                                       ["method_name", "software_name", "parameters", "evidence_text", "paper_title"])
            conn.commit()
        finally:
            conn.close()

    def _create_fts_triggers(self, conn, table: str, fts_table: str, columns: list[str]) -> None:
        cols = ", ".join(columns)
        new_cols = ", ".join(f"new.{c}" for c in columns)
        old_cols = ", ".join(f"old.{c}" for c in columns)

        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table} BEGIN
                INSERT INTO {fts_table}(rowid, {cols}) VALUES (new.id, {new_cols});
            END
        """)
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table} BEGIN
                INSERT INTO {fts_table}({fts_table}, rowid, {cols}) VALUES ('delete', old.id, {old_cols});
            END
        """)
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table} BEGIN
                INSERT INTO {fts_table}({fts_table}, rowid, {cols}) VALUES ('delete', old.id, {old_cols});
                INSERT INTO {fts_table}(rowid, {cols}) VALUES (new.id, {new_cols});
            END
        """)

    def _init_protocols_db(self) -> None:
        conn = sqlite3.connect(str(self._prot_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS protocols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protocol_id TEXT UNIQUE NOT NULL,
                    source TEXT,
                    external_id TEXT,
                    title TEXT,
                    url TEXT,
                    domain TEXT,
                    subdomain TEXT,
                    authors TEXT,
                    last_author TEXT,
                    summary TEXT,
                    materials TEXT,
                    reagents TEXT,
                    instruments TEXT,
                    software TEXT,
                    databases TEXT,
                    steps TEXT,
                    tags TEXT,
                    access TEXT,
                    fetched_at TEXT,
                    missing_information TEXT,
                    evidence TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS protocols_fts USING fts5(
                    title, summary, domain, subdomain, tags,
                    authors, materials, reagents, steps,
                    content='protocols',
                    content_rowid='id'
                )
            """)
            self._create_fts_triggers(conn, "protocols", "protocols_fts",
                                       ["title", "summary", "domain", "subdomain", "tags",
                                        "authors", "materials", "reagents", "steps"])
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Literature: articles
    # ------------------------------------------------------------------

    def add_literature_article(self, article: LiteratureArticle) -> bool:
        """Add a literature article. Returns True if inserted, False if duplicate."""
        conn = sqlite3.connect(str(self._lit_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO articles
                    (pmid, doi, title, journal, year, abstract, authors,
                     last_author, corresponding_author, mesh_terms,
                     publication_types, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.pmid,
                    article.doi,
                    article.title,
                    article.journal,
                    article.year,
                    article.abstract,
                    json.dumps(article.authors, ensure_ascii=False),
                    article.last_author,
                    article.corresponding_author,
                    json.dumps(article.mesh_terms, ensure_ascii=False),
                    json.dumps(article.publication_types, ensure_ascii=False),
                    article.fetched_at,
                ),
            )
            inserted = cursor.rowcount > 0
            conn.commit()
            # JSONL backup
            if inserted:
                self._append_jsonl(self._lit_jsonl_path, article.model_dump())
            return inserted
        finally:
            conn.close()

    def add_literature_review_row(self, row: LiteratureReviewRow) -> bool:
        """Add a literature review row. Returns True if inserted."""
        conn = sqlite3.connect(str(self._lit_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                INSERT INTO review_rows
                    (pmid, method_name, software_name, parameters, lab_info,
                     paper_title, journal, doi, year, evidence_text, missing_fields)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.pmid,
                    row.method_name,
                    row.software_name,
                    row.parameters,
                    row.lab_info,
                    row.paper_title,
                    row.journal,
                    row.doi,
                    row.year,
                    row.evidence_text,
                    json.dumps(row.missing_fields, ensure_ascii=False),
                ),
            )
            conn.commit()
            self._append_jsonl(self._lit_jsonl_path, row.model_dump())
            return True
        finally:
            conn.close()

    def search_literature(self, query: str, top_k: int = 10) -> list[dict]:
        """Full-text search across literature articles using FTS5."""
        conn = sqlite3.connect(str(self._lit_db_path))
        try:
            # Try FTS5 search on articles
            fts_query = self._to_fts_query(query)
            cursor = conn.execute(
                """
                SELECT a.pmid, a.doi, a.title, a.journal, a.year, a.abstract,
                       a.authors, a.last_author, a.corresponding_author,
                       a.mesh_terms, a.publication_types, a.fetched_at,
                       articles_fts.rank
                FROM articles_fts
                JOIN articles a ON articles_fts.rowid = a.id
                WHERE articles_fts MATCH ?
                ORDER BY articles_fts.rank
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            results = []
            for row in cursor:
                results.append({
                    "pmid": row[0], "doi": row[1], "title": row[2],
                    "journal": row[3], "year": row[4], "abstract": row[5],
                    "authors": self._parse_json(row[6]),
                    "last_author": row[7], "corresponding_author": row[8],
                    "mesh_terms": self._parse_json(row[9]),
                    "publication_types": self._parse_json(row[10]),
                    "fetched_at": row[11],
                    "score": row[12] if len(row) > 12 else None,
                })
            # If FTS5 returned nothing, fall back to LIKE search
            if not results:
                results = self._fallback_like_search(conn, "articles", query, top_k)
            return results
        finally:
            conn.close()

    def search_literature_rows(self, query: str, top_k: int = 10) -> list[dict]:
        """Full-text search across literature review rows using FTS5."""
        conn = sqlite3.connect(str(self._lit_db_path))
        try:
            fts_query = self._to_fts_query(query)
            cursor = conn.execute(
                """
                SELECT r.pmid, r.method_name, r.software_name, r.parameters,
                       r.lab_info, r.paper_title, r.journal, r.doi, r.year,
                       r.evidence_text, r.missing_fields,
                       review_rows_fts.rank
                FROM review_rows_fts
                JOIN review_rows r ON review_rows_fts.rowid = r.id
                WHERE review_rows_fts MATCH ?
                ORDER BY review_rows_fts.rank
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            results = []
            for row in cursor:
                results.append({
                    "pmid": row[0], "method_name": row[1], "software_name": row[2],
                    "parameters": row[3], "lab_info": row[4], "paper_title": row[5],
                    "journal": row[6], "doi": row[7], "year": row[8],
                    "evidence_text": row[9],
                    "missing_fields": self._parse_json(row[10]),
                    "score": row[11] if len(row) > 11 else None,
                })
            if not results:
                results = self._fallback_like_search(conn, "review_rows", query, top_k)
            return results
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def add_protocol(self, protocol: BioProtocolRecord) -> bool:
        """Add a protocol record. Returns True if inserted, False if duplicate."""
        conn = sqlite3.connect(str(self._prot_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            steps_json = json.dumps(
                [s.model_dump() if hasattr(s, 'model_dump') else s for s in protocol.steps],
                ensure_ascii=False
            )
            evidence_json = json.dumps(
                [e.model_dump() if hasattr(e, 'model_dump') else e for e in protocol.evidence],
                ensure_ascii=False
            )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO protocols
                    (protocol_id, source, external_id, title, url, domain, subdomain,
                     authors, last_author, summary, materials, reagents, instruments,
                     software, databases, steps, tags, access, fetched_at,
                     missing_information, evidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    protocol.protocol_id,
                    protocol.source,
                    protocol.external_id,
                    protocol.title,
                    protocol.url,
                    protocol.domain,
                    protocol.subdomain,
                    json.dumps(protocol.authors, ensure_ascii=False),
                    protocol.last_author,
                    protocol.summary,
                    json.dumps(protocol.materials, ensure_ascii=False),
                    json.dumps(protocol.reagents, ensure_ascii=False),
                    json.dumps(protocol.instruments, ensure_ascii=False),
                    json.dumps(protocol.software, ensure_ascii=False),
                    json.dumps(protocol.databases, ensure_ascii=False),
                    steps_json,
                    json.dumps(protocol.tags, ensure_ascii=False),
                    protocol.access,
                    protocol.fetched_at,
                    json.dumps(protocol.missing_information, ensure_ascii=False),
                    evidence_json,
                ),
            )
            inserted = cursor.rowcount > 0
            conn.commit()
            if inserted:
                self._append_jsonl(self._prot_jsonl_path, protocol.model_dump())
            return inserted
        finally:
            conn.close()

    def search_protocols(self, query: str, top_k: int = 10) -> list[dict]:
        """Full-text search across protocol records using FTS5."""
        conn = sqlite3.connect(str(self._prot_db_path))
        try:
            fts_query = self._to_fts_query(query)
            cursor = conn.execute(
                """
                SELECT p.protocol_id, p.source, p.external_id, p.title, p.url,
                       p.domain, p.subdomain, p.authors, p.last_author, p.summary,
                       p.materials, p.reagents, p.instruments, p.software,
                       p.databases, p.steps, p.tags, p.access, p.fetched_at,
                       p.missing_information,
                       protocols_fts.rank
                FROM protocols_fts
                JOIN protocols p ON protocols_fts.rowid = p.id
                WHERE protocols_fts MATCH ?
                ORDER BY protocols_fts.rank
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            results = []
            for row in cursor:
                results.append({
                    "protocol_id": row[0], "source": row[1],
                    "external_id": row[2], "title": row[3], "url": row[4],
                    "domain": row[5], "subdomain": row[6],
                    "authors": self._parse_json(row[7]),
                    "last_author": row[8], "summary": row[9],
                    "materials": self._parse_json(row[10]),
                    "reagents": self._parse_json(row[11]),
                    "instruments": self._parse_json(row[12]),
                    "software": self._parse_json(row[13]),
                    "databases": self._parse_json(row[14]),
                    "steps": self._parse_json(row[15]),
                    "tags": self._parse_json(row[16]),
                    "access": row[17], "fetched_at": row[18],
                    "missing_information": self._parse_json(row[19]),
                    "score": row[20] if len(row) > 20 else None,
                })
            if not results:
                results = self._fallback_like_search(conn, "protocols", query, top_k)
            return results
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return aggregate statistics for both RAG stores."""
        stats = {
            "literature": {"articles": 0, "review_rows": 0, "db_path": str(self._lit_db_path)},
            "protocols": {"records": 0, "db_path": str(self._prot_db_path)},
        }
        # Literature
        if self._lit_db_path.exists():
            try:
                conn = sqlite3.connect(str(self._lit_db_path))
                stats["literature"]["articles"] = conn.execute(
                    "SELECT COUNT(*) FROM articles"
                ).fetchone()[0]
                stats["literature"]["review_rows"] = conn.execute(
                    "SELECT COUNT(*) FROM review_rows"
                ).fetchone()[0]
                conn.close()
            except Exception as e:
                logger.warning("Failed to get literature stats: %s", e)

        # Protocols
        if self._prot_db_path.exists():
            try:
                conn = sqlite3.connect(str(self._prot_db_path))
                stats["protocols"]["records"] = conn.execute(
                    "SELECT COUNT(*) FROM protocols"
                ).fetchone()[0]
                # Per-source breakdown
                source_counts = conn.execute(
                    "SELECT source, COUNT(*) FROM protocols GROUP BY source"
                ).fetchall()
                stats["protocols"]["by_source"] = dict(source_counts)
                conn.close()
            except Exception as e:
                logger.warning("Failed to get protocol stats: %s", e)

        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_fts_query(self, query: str) -> str:
        """Convert a user query to an FTS5-compatible query string."""
        # Split into terms, wrap each in quotes for exact matching, OR them together
        terms = [t.strip() for t in query.split() if t.strip()]
        if not terms:
            return "*"
        # Escape double quotes inside terms
        escaped = [t.replace('"', '""') for t in terms]
        return " OR ".join(f'"{t}"' for t in escaped)

    def _fallback_like_search(self, conn, table: str, query: str, top_k: int) -> list[dict]:
        """Fallback LIKE-based search when FTS5 returns no results."""
        like_term = f"%{query}%"
        if table == "articles":
            cursor = conn.execute(
                """SELECT pmid, doi, title, journal, year, abstract,
                          authors, last_author, corresponding_author,
                          mesh_terms, publication_types, fetched_at
                   FROM articles
                   WHERE title LIKE ? OR abstract LIKE ? OR journal LIKE ?
                   LIMIT ?""",
                (like_term, like_term, like_term, top_k),
            )
            results = []
            for row in cursor:
                results.append({
                    "pmid": row[0], "doi": row[1], "title": row[2],
                    "journal": row[3], "year": row[4], "abstract": row[5],
                    "authors": self._parse_json(row[6]),
                    "last_author": row[7], "corresponding_author": row[8],
                    "mesh_terms": self._parse_json(row[9]),
                    "publication_types": self._parse_json(row[10]),
                    "fetched_at": row[11], "score": None,
                })
            return results
        elif table == "review_rows":
            cursor = conn.execute(
                """SELECT pmid, method_name, software_name, parameters,
                          lab_info, paper_title, journal, doi, year,
                          evidence_text, missing_fields
                   FROM review_rows
                   WHERE method_name LIKE ? OR software_name LIKE ? OR evidence_text LIKE ?
                   LIMIT ?""",
                (like_term, like_term, like_term, top_k),
            )
            results = []
            for row in cursor:
                results.append({
                    "pmid": row[0], "method_name": row[1], "software_name": row[2],
                    "parameters": row[3], "lab_info": row[4], "paper_title": row[5],
                    "journal": row[6], "doi": row[7], "year": row[8],
                    "evidence_text": row[9],
                    "missing_fields": self._parse_json(row[10]),
                    "score": None,
                })
            return results
        elif table == "protocols":
            cursor = conn.execute(
                """SELECT protocol_id, source, external_id, title, url,
                          domain, subdomain, authors, last_author, summary,
                          materials, reagents, instruments, software,
                          databases, steps, tags, access, fetched_at,
                          missing_information
                   FROM protocols
                   WHERE title LIKE ? OR summary LIKE ? OR domain LIKE ? OR tags LIKE ?
                   LIMIT ?""",
                (like_term, like_term, like_term, like_term, top_k),
            )
            results = []
            for row in cursor:
                results.append({
                    "protocol_id": row[0], "source": row[1],
                    "external_id": row[2], "title": row[3], "url": row[4],
                    "domain": row[5], "subdomain": row[6],
                    "authors": self._parse_json(row[7]),
                    "last_author": row[8], "summary": row[9],
                    "materials": self._parse_json(row[10]),
                    "reagents": self._parse_json(row[11]),
                    "instruments": self._parse_json(row[12]),
                    "software": self._parse_json(row[13]),
                    "databases": self._parse_json(row[14]),
                    "steps": self._parse_json(row[15]),
                    "tags": self._parse_json(row[16]),
                    "access": row[17], "fetched_at": row[18],
                    "missing_information": self._parse_json(row[19]),
                    "score": None,
                })
            return results
        return []

    @staticmethod
    def _parse_json(value: str | None) -> list | dict:
        if not value:
            return []
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value if value else []

    def _append_jsonl(self, path: Path, data: dict) -> None:
        """Append one record to a JSONL backup file."""
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write JSONL backup to %s: %s", path, e)


def default_rag_store(root: str | Path | None = None) -> SQLiteRagStore:
    """Get or create the default RAG store instance."""
    store = SQLiteRagStore(root=root)
    store.init_db()
    return store
