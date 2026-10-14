from scicore.bio.cli import build_arg_parser
from scicore.agent.skill_router import SkillRoute, SkillRouter, SkillTier
from scicore.kernel.features.research_graph import ResearchGraphProjector
from scicore.kernel.memory_pipeline import MemoryPipeline
from scicore.kernel.session_store import SessionStore


def _decision_memory():
    return {
        "memory_id": "mem-decision-1",
        "memory_type": "decision",
        "statement": "决定使用 DESeq2，并生成 results.csv。",
        "user_scope": "user-a",
        "project_scope": "project-rna",
        "reliability": 0.9,
        "created_at": "2026-07-30T08:00:00+00:00",
        "valid_to": "2026-08-30T08:00:00+00:00",
        "evidence": {
            "source_uri": "session://one/messages/2-3#sha256=" + "a" * 64,
            "source_sha256": "a" * 64,
            "session_id": "one",
        },
    }


def test_research_graph_projects_entities_event_effective_time_and_evidence():
    graph = ResearchGraphProjector().project(_decision_memory())

    assert graph["event"]["event_type"] == "decision"
    assert graph["event"]["valid_from"] == "2026-07-30T08:00:00+00:00"
    assert graph["event"]["valid_to"] == "2026-08-30T08:00:00+00:00"
    assert {item["entity_type"] for item in graph["entities"]} >= {
        "user",
        "project",
        "artifact",
        "method_or_decision",
    }
    assert graph["evidence_links"][0]["source_uri"].startswith("session://one/")
    assert graph["evidence_links"][0]["evidence_role"] == "supports"


def test_research_graph_accepts_explicit_causal_event_relation():
    memory = _decision_memory()
    memory["graph"] = {
        "event_type": "recovery",
        "event_relations": [
            {
                "predicate": "caused_by",
                "object_event_id": "evtkg_previous",
                "confidence": 0.95,
            }
        ],
    }

    graph = ResearchGraphProjector().project(memory)

    assert graph["event"]["event_type"] == "recovery"
    assert graph["event_relations"][0]["predicate"] == "caused_by"
    assert graph["event_relations"][0]["object_event_id"] == "evtkg_previous"


def test_preference_does_not_become_a_spurious_decision_entity():
    memory = _decision_memory()
    memory.update(
        {
            "memory_type": "user_preference",
            "statement": "希望报告始终使用中文。",
        }
    )

    graph = ResearchGraphProjector().project(memory)

    assert "method_or_decision" not in {
        item["entity_type"] for item in graph["entities"]
    }


def test_async_memory_pipeline_projects_postgres_candidates_to_graph(tmp_path):
    class _GraphBackend:
        def __init__(self):
            self.memories = []
            self.graph_memories = []

        def upsert(self, memory):
            self.memories.append(memory)

        def upsert_graph_memory(self, memory):
            self.graph_memories.append(memory)

    store = SessionStore(tmp_path / "sessions")
    session = {
        "id": "graph-session",
        "created_at": "2026-07-30T00:00:00+00:00",
        "workspace_root": str(tmp_path),
        "memory": {},
        "history": [
            {"role": "assistant", "content": "决定使用 DESeq2 生成 results.csv。"}
        ],
    }
    store.save(session)
    backend = _GraphBackend()
    pipeline = MemoryPipeline(store, backend=backend)
    pipeline.capture(
        session, metadata={"user_scope": "user-a", "project_scope": "project-rna"}
    )

    assert pipeline.drain()["processed"] == 1
    assert len(backend.memories) == len(backend.graph_memories) == 1


def test_skill_router_uses_catalog_recall_metadata_rerank_and_boundaries():
    router = SkillRouter()
    router.register(
        SkillRoute(
            slug="bulk-rna",
            title="Bulk RNA-seq differential expression",
            tier=SkillTier.CORE,
            priority=90,
            triggers=["RNA-seq", "差异表达"],
            tags=["transcriptomics"],
            applicable_when=["bulk count matrix"],
            boundaries=["single-cell"],
            examples=["compare treatment and control counts"],
        )
    )
    router.register(
        SkillRoute(
            slug="unrelated",
            title="Protein structure rendering",
            triggers=["protein structure"],
            tags=["visualization"],
        )
    )

    rows = router.route("compare treatment and control counts for RNA-seq")

    assert rows and rows[0].skill.slug == "bulk-rna"
    assert router.last_route_diagnostics["strategy"] == (
        "catalog_recall_then_metadata_rerank"
    )
    assert router.last_route_diagnostics["candidate_count"] == 1
    assert router.route("single-cell RNA-seq differential expression") == []


def test_research_graph_query_options_are_exposed_by_cli():
    args = build_arg_parser().parse_args(
        ["--memory-timeline", "user-a", "project-rna"]
    )

    assert args.memory_timeline == ["user-a", "project-rna"]
