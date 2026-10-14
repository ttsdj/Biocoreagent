"""Temporal, evidence-backed research knowledge graph.

The graph is deliberately stored in PostgreSQL beside distilled long-term
memory.  It models research entities, events, event-to-event relations and
immutable evidence links without requiring a separate graph database.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

_ARTIFACT = re.compile(
    r"(?P<path>[\w./\\-]+\.(?:csv|tsv|xlsx|json|parquet|h5ad|rds|pdf|svg|png|py|r|ipynb))",
    re.IGNORECASE,
)
_DECISION_OBJECT = re.compile(
    r"(?:决定(?:采用|使用|选择)?|采用|切换为|选择|decided to use|adopted?)"
    r"\s*(?P<object>[^。；;\n]{2,160})",
    re.IGNORECASE,
)


def _stable_id(prefix: str, *parts: object) -> str:
    canonical = "\0".join(str(part) for part in parts)
    return prefix + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class ResearchGraphProjector:
    """Project a distilled memory into entities, an event and evidence links.

    Callers may provide an explicit ``graph`` object on a memory candidate for
    high-fidelity relations.  The deterministic fallback still creates useful
    project/user, artifact and decision-object entities.
    """

    version = "research_graph_v1"

    def project(self, memory):
        memory = dict(memory)
        explicit = dict(memory.get("graph") or {})
        statement = str(memory.get("statement", "")).strip()
        if not statement:
            raise ValueError("research graph projection requires a statement")
        user_scope = str(memory.get("user_scope", ""))
        project_scope = str(memory.get("project_scope", ""))
        evidence = dict(memory.get("evidence") or {})
        valid_from = str(
            explicit.get("valid_from")
            or memory.get("valid_from")
            or memory.get("created_at")
            or datetime.now(timezone.utc).isoformat()
        )
        valid_to = explicit.get("valid_to", memory.get("valid_to"))
        event_id = str(
            explicit.get("event_id")
            or _stable_id("evtkg_", memory.get("memory_id", ""), statement, project_scope)
        )
        event_type = str(
            explicit.get("event_type")
            or {
                "decision": "decision",
                "user_preference": "preference",
                "exception": "exception",
                "recovery": "recovery",
            }.get(str(memory.get("memory_type", "")), "observation")
        )

        entities = []
        participants = []

        def add_entity(entity_type, canonical_name, role, attributes=None):
            canonical_name = str(canonical_name).strip()
            if not canonical_name:
                return ""
            entity_id = _stable_id(
                "ent_", entity_type, canonical_name.lower(), user_scope, project_scope
            )
            if entity_id not in {item["entity_id"] for item in entities}:
                entities.append(
                    {
                        "entity_id": entity_id,
                        "entity_type": str(entity_type),
                        "canonical_name": canonical_name,
                        "attributes": dict(attributes or {}),
                    }
                )
            if (event_id, entity_id, role) not in {
                (item["event_id"], item["entity_id"], item["role"])
                for item in participants
            }:
                participants.append(
                    {"event_id": event_id, "entity_id": entity_id, "role": str(role)}
                )
            return entity_id

        if user_scope:
            add_entity("user", user_scope, "actor")
        if project_scope:
            add_entity("project", project_scope, "context")
        for match in _ARTIFACT.finditer(statement):
            add_entity("artifact", match.group("path"), "artifact")
        decision_object = (
            _DECISION_OBJECT.search(statement)
            if str(memory.get("memory_type", "")) == "decision"
            else None
        )
        if decision_object:
            add_entity("method_or_decision", decision_object.group("object"), "object")
        for item in explicit.get("entities", []):
            item = dict(item)
            add_entity(
                item.get("entity_type", "concept"),
                item.get("canonical_name", item.get("name", "")),
                item.get("role", "related"),
                item.get("attributes"),
            )

        event_relations = []
        for relation in explicit.get("event_relations", []):
            relation = dict(relation)
            other = str(relation.get("object_event_id", "")).strip()
            predicate = str(relation.get("predicate", "")).strip()
            if other and predicate:
                event_relations.append(
                    {
                        "relation_id": _stable_id("evrel_", event_id, predicate, other),
                        "subject_event_id": event_id,
                        "predicate": predicate,
                        "object_event_id": other,
                        "confidence": float(relation.get("confidence", 0.8)),
                    }
                )
        supersedes = str(memory.get("supersedes", "")).strip()
        if supersedes:
            event_relations.append(
                {
                    "relation_id": _stable_id("evrel_", event_id, "supersedes", supersedes),
                    "subject_event_id": event_id,
                    "predicate": "supersedes",
                    "object_event_id": supersedes,
                    "confidence": float(memory.get("reliability", 0.65)),
                }
            )

        source_uri = str(evidence.get("source_uri", "")).strip()
        evidence_links = []
        if source_uri:
            evidence_links.append(
                {
                    "evidence_id": _stable_id("evi_", event_id, source_uri),
                    "target_kind": "event",
                    "target_id": event_id,
                    "source_uri": source_uri,
                    "source_sha256": str(evidence.get("source_sha256", "")),
                    "evidence_role": str(explicit.get("evidence_role", "supports")),
                }
            )

        return {
            "projection_version": self.version,
            "user_scope": user_scope,
            "project_scope": project_scope,
            "event": {
                "event_id": event_id,
                "event_type": event_type,
                "statement": statement,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence": float(memory.get("reliability", 0.65)),
                "attributes": dict(explicit.get("attributes") or {}),
            },
            "entities": entities,
            "participants": participants,
            "event_relations": event_relations,
            "evidence_links": evidence_links,
        }


class PostgresResearchGraphStore:
    """PostgreSQL implementation of the temporal research graph."""

    def __init__(self, dsn, *, projector=None):
        self.dsn = str(dsn)
        self.projector = projector or ResearchGraphProjector()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("psycopg is required for the PostgreSQL research graph") from exc
        return psycopg.connect(self.dsn)

    def initialize(self):
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_entities (
                    entity_id TEXT PRIMARY KEY,
                    user_scope TEXT NOT NULL DEFAULT '',
                    project_scope TEXT NOT NULL DEFAULT '',
                    entity_type TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    attributes_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_scope, project_scope, entity_type, canonical_name)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_events (
                    event_id TEXT PRIMARY KEY,
                    user_scope TEXT NOT NULL DEFAULT '',
                    project_scope TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    valid_to TIMESTAMPTZ,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.65,
                    attributes_json JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_event_participants (
                    event_id TEXT NOT NULL REFERENCES research_events(event_id),
                    entity_id TEXT NOT NULL REFERENCES research_entities(entity_id),
                    role TEXT NOT NULL,
                    PRIMARY KEY(event_id, entity_id, role)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_event_relations (
                    relation_id TEXT PRIMARY KEY,
                    subject_event_id TEXT NOT NULL REFERENCES research_events(event_id),
                    predicate TEXT NOT NULL,
                    object_event_id TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.65
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_evidence_links (
                    evidence_id TEXT PRIMARY KEY,
                    target_kind TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL DEFAULT '',
                    evidence_role TEXT NOT NULL DEFAULT 'supports',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(target_kind, target_id, source_uri)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_events_timeline "
                "ON research_events(user_scope, project_scope, valid_from)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_participants_entity "
                "ON research_event_participants(entity_id, event_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_event_relations_object "
                "ON research_event_relations(object_event_id, predicate)"
            )
        return self

    def upsert_memory(self, memory):
        graph = self.projector.project(memory)
        self.initialize()
        event = graph["event"]
        with self._connect() as connection, connection.cursor() as cursor:
            for entity in graph["entities"]:
                cursor.execute(
                    """
                    INSERT INTO research_entities (
                        entity_id, user_scope, project_scope, entity_type,
                        canonical_name, attributes_json
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        attributes_json = EXCLUDED.attributes_json
                    """,
                    (
                        entity["entity_id"],
                        graph["user_scope"],
                        graph["project_scope"],
                        entity["entity_type"],
                        entity["canonical_name"],
                        json.dumps(entity["attributes"], ensure_ascii=False),
                    ),
                )
            cursor.execute(
                """
                INSERT INTO research_events (
                    event_id, user_scope, project_scope, event_type, statement,
                    valid_from, valid_to, confidence, attributes_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT(event_id) DO UPDATE SET
                    statement = EXCLUDED.statement,
                    valid_from = EXCLUDED.valid_from,
                    valid_to = EXCLUDED.valid_to,
                    confidence = EXCLUDED.confidence,
                    attributes_json = EXCLUDED.attributes_json
                """,
                (
                    event["event_id"],
                    graph["user_scope"],
                    graph["project_scope"],
                    event["event_type"],
                    event["statement"],
                    event["valid_from"],
                    event["valid_to"],
                    event["confidence"],
                    json.dumps(event["attributes"], ensure_ascii=False),
                ),
            )
            for participant in graph["participants"]:
                cursor.execute(
                    """
                    INSERT INTO research_event_participants(event_id, entity_id, role)
                    VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
                    """,
                    (
                        participant["event_id"],
                        participant["entity_id"],
                        participant["role"],
                    ),
                )
            for relation in graph["event_relations"]:
                # The referenced historical event may not have been projected
                # yet.  Keep ingestion retryable instead of inventing a node.
                cursor.execute(
                    "SELECT 1 FROM research_events WHERE event_id = %s",
                    (relation["object_event_id"],),
                )
                if cursor.fetchone():
                    cursor.execute(
                        """
                        INSERT INTO research_event_relations(
                            relation_id, subject_event_id, predicate,
                            object_event_id, confidence
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT(relation_id) DO UPDATE SET
                            confidence = EXCLUDED.confidence
                        """,
                        (
                            relation["relation_id"],
                            relation["subject_event_id"],
                            relation["predicate"],
                            relation["object_event_id"],
                            relation["confidence"],
                        ),
                    )
            for evidence in graph["evidence_links"]:
                cursor.execute(
                    """
                    INSERT INTO research_evidence_links(
                        evidence_id, target_kind, target_id, source_uri,
                        source_sha256, evidence_role
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(evidence_id) DO NOTHING
                    """,
                    (
                        evidence["evidence_id"],
                        evidence["target_kind"],
                        evidence["target_id"],
                        evidence["source_uri"],
                        evidence["source_sha256"],
                        evidence["evidence_role"],
                    ),
                )
        return event["event_id"]

    def timeline(self, *, user_scope="", project_scope="", entity_id="", limit=100):
        where = []
        values = []
        if user_scope:
            where.append("event.user_scope = %s")
            values.append(str(user_scope))
        else:
            where.append("event.user_scope = ''")
        if project_scope:
            where.append("event.project_scope = %s")
            values.append(str(project_scope))
        else:
            where.append("event.project_scope = ''")
        join = ""
        if entity_id:
            join = (
                " JOIN research_event_participants participant "
                "ON participant.event_id = event.event_id "
            )
            where.append("participant.entity_id = %s")
            values.append(str(entity_id))
        values.append(max(0, int(limit)))
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event.event_id, event.event_type, event.statement,
                       event.valid_from, event.valid_to, event.confidence
                FROM research_events event
                """
                + join
                + " WHERE "
                + " AND ".join(where)
                + " ORDER BY event.valid_from, event.recorded_at LIMIT %s",
                tuple(values),
            )
            columns = [item.name for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def explain_event(self, event_id):
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event.event_id, event.event_type, event.statement,
                       event.valid_from, event.valid_to, event.confidence,
                       COALESCE(
                           jsonb_agg(DISTINCT jsonb_build_object(
                               'source_uri', evidence.source_uri,
                               'source_sha256', evidence.source_sha256,
                               'role', evidence.evidence_role
                           )) FILTER (WHERE evidence.evidence_id IS NOT NULL),
                           '[]'::jsonb
                       ) AS evidence
                FROM research_events event
                LEFT JOIN research_evidence_links evidence
                  ON evidence.target_kind = 'event'
                 AND evidence.target_id = event.event_id
                WHERE event.event_id = %s
                GROUP BY event.event_id
                """,
                (str(event_id),),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return dict(zip([item.name for item in cursor.description], row))

    def causal_chain(self, event_id, *, max_depth=8):
        """Return caused-by/recovery/supersession neighbours via recursive SQL."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                WITH RECURSIVE chain AS (
                    SELECT relation_id, subject_event_id, predicate,
                           object_event_id, 1 AS depth
                    FROM research_event_relations
                    WHERE subject_event_id = %s
                      AND predicate IN ('caused_by', 'recovers', 'supersedes')
                    UNION ALL
                    SELECT relation.relation_id, relation.subject_event_id,
                           relation.predicate, relation.object_event_id,
                           chain.depth + 1
                    FROM research_event_relations relation
                    JOIN chain ON relation.subject_event_id = chain.object_event_id
                    WHERE chain.depth < %s
                      AND relation.predicate IN ('caused_by', 'recovers', 'supersedes')
                )
                SELECT chain.depth, chain.predicate, event.event_id,
                       event.event_type, event.statement, event.valid_from
                FROM chain
                JOIN research_events event ON event.event_id = chain.object_event_id
                ORDER BY chain.depth
                """,
                (str(event_id), max(1, int(max_depth))),
            )
            columns = [item.name for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
