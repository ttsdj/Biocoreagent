"""统一知识库工具 — 将 UnifiedKnowledgeBase 暴露为 Agent 可调用的工具。"""

from __future__ import annotations

import json

from .base import Tool
from ..knowledge_base import get_knowledge_base, KnowledgeSource


class KBCrossSearchTool(Tool):
    """跨知识库统一查询工具。

    同时搜索 Wiki、RAG（文献+协议）、Skill，返回合并去重的结果。
    """

    name = "kb_cross_search"
    description = (
        "跨知识库统一查询：同时搜索 Wiki 经验、RAG 文献、RAG 协议和 Skill 流程，"
        "返回合并去重并按相关性排序的结果。用于在多个知识源中查找相关信息。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询文本（中文或英文）",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的最大结果数",
                "default": 10,
            },
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": KnowledgeSource.all_sources()},
                "description": "限定搜索的知识源。不提供则搜索全部。可选: wiki, rag_literature, rag_protocol, skill",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, top_k: int = 10, sources: list[str] | None = None) -> str:
        kb = get_knowledge_base()
        results = kb.cross_search(query=query, top_k=top_k, sources=sources)

        if not results:
            return json.dumps({
                "found": False,
                "query": query,
                "message": "未在任何知识库中找到相关内容。",
                "results": [],
            }, ensure_ascii=False, indent=2)

        # 按来源分组统计
        source_counts: dict[str, int] = {}
        for r in results:
            source_counts[r.source_label] = source_counts.get(r.source_label, 0) + 1

        return json.dumps({
            "found": True,
            "query": query,
            "total": len(results),
            "source_counts": source_counts,
            "results": [r.to_dict() for r in results],
        }, ensure_ascii=False, indent=2)


class KBCrossEnhanceTool(Tool):
    """交叉增强工具。

    从一条知识出发，找到其他知识源中的关联内容。
    """

    name = "kb_cross_enhance"
    description = (
        "交叉增强：给定一条知识的 ID，自动在其他知识库中找到关联内容。"
        "例如，从一条 Wiki 经验出发，找到相关的 Skill、RAG 协议和文献。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "wiki_slug": {
                "type": "string",
                "description": "Wiki 条目 slug（可选）",
            },
            "skill_slug": {
                "type": "string",
                "description": "Skill slug（可选）",
            },
            "rag_protocol_id": {
                "type": "string",
                "description": "RAG 协议 ID（可选）",
            },
        },
    }

    def execute(
        self,
        wiki_slug: str | None = None,
        skill_slug: str | None = None,
        rag_protocol_id: str | None = None,
    ) -> str:
        kb = get_knowledge_base()
        result = kb.cross_enhance(
            wiki_slug=wiki_slug,
            skill_slug=skill_slug,
            rag_protocol_id=rag_protocol_id,
        )

        # Count related items
        related_counts = {
            "wiki": len(result.get("related_wiki", [])),
            "skills": len(result.get("related_skills", [])),
            "protocols": len(result.get("related_protocols", [])),
            "literature": len(result.get("related_literature", [])),
        }

        return json.dumps({
            "source_item": result.get("source_item"),
            "related_counts": related_counts,
            "related_wiki": result.get("related_wiki", []),
            "related_skills": result.get("related_skills", []),
            "related_protocols": result.get("related_protocols", []),
            "related_literature": result.get("related_literature", []),
        }, ensure_ascii=False, indent=2)


class KBSuggestSkillTool(Tool):
    """Wiki 经验自动提纯为 Skill 的工具。

    检测 Wiki 中重复出现 ≥3 次的主题，建议转化为可复用的 Skill。
    """

    name = "kb_suggest_skill"
    description = (
        "检测 Wiki 经验库中重复出现的主题（≥3 次），建议将这些经验提纯为可复用的 Skill 流程。"
        "这是知识库自进化机制的核心：经验积累 → 模式识别 → 流程沉淀。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "min_occurrence": {
                "type": "integer",
                "description": "最少出现次数阈值（默认 3）",
                "default": 3,
            },
        },
    }

    def execute(self, min_occurrence: int = 3) -> str:
        kb = get_knowledge_base()
        suggestions = kb.suggest_skill_from_wiki(min_occurrence=min_occurrence)

        if not suggestions:
            return json.dumps({
                "found": False,
                "message": (
                    f"未找到重复 ≥{min_occurrence} 次的 Wiki 经验组。"
                    "继续积累经验后，系统会自动识别可提纯的模式。"
                ),
                "suggestions": [],
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "found": True,
            "total_suggestions": len(suggestions),
            "message": f"发现 {len(suggestions)} 组可提纯的 Wiki 经验",
            "suggestions": suggestions,
        }, ensure_ascii=False, indent=2)
