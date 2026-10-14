"""统一知识库 — Wiki ↔ RAG ↔ Skill 深度整合.

实现三个知识存储的统一查询、交叉检索和自动转化：
  - Wiki（经验知识）：会话中积累的程序性经验、情景记忆、用户画像
  - RAG（文献/协议数据）：PubMed 文献、Protocols.io 协议的结构化数据
  - Skill（流程知识）：可复用的工作流程和最佳实践

核心机制：
  1. 统一查询：一次查询同时搜索三个知识库，合并去重排序
  2. 交叉增强：RAG 数据补充 Wiki 经验，Wiki 经验指导 Skill 选择
  3. 自动提纯：Wiki 中重复出现 ≥3 次的主题自动提示转化为 Skill
  4. 知识图谱：生成跨三个库的知识关联上下文

Reference: SciCoreAgent 需求文档, Section 5 (自进化个人知识库体系).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Knowledge Source Enum
# ---------------------------------------------------------------------------


class KnowledgeSource:
    """知识来源枚举。"""

    WIKI = "wiki"
    RAG_LITERATURE = "rag_literature"
    RAG_PROTOCOL = "rag_protocol"
    SKILL = "skill"

    @classmethod
    def all_sources(cls) -> list[str]:
        return [cls.WIKI, cls.RAG_LITERATURE, cls.RAG_PROTOCOL, cls.SKILL]

    @classmethod
    def labels(cls) -> dict[str, str]:
        return {
            cls.WIKI: "📝 经验知识 (Wiki)",
            cls.RAG_LITERATURE: "📄 文献数据 (RAG)",
            cls.RAG_PROTOCOL: "🧪 协议数据 (RAG)",
            cls.SKILL: "🔧 流程技能 (Skill)",
        }


# ---------------------------------------------------------------------------
# Unified Search Result
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeItem:
    """统一的知识条目。"""

    source: str  # wiki / rag_literature / rag_protocol / skill
    source_label: str
    id: str
    title: str
    content: str
    score: float  # 0.0 - 1.0 相关性分数
    tags: list[str] = field(default_factory=list)
    url: str = ""  # RAG 条目可能有 URL
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_label": self.source_label,
            "id": self.id,
            "title": self.title,
            "content": self.content[:500],  # 截断
            "score": self.score,
            "tags": self.tags,
            "url": self.url,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Unified Knowledge Base
# ---------------------------------------------------------------------------


class UnifiedKnowledgeBase:
    """统一知识库 — 跨 Wiki、RAG、Skill 的统一查询接口。

    Usage::

        kb = UnifiedKnowledgeBase()
        results = kb.cross_search("RNA-seq 差异表达 STAR DESeq2", top_k=10)
        for item in results:
            print(f"[{item.source_label}] {item.title} (score={item.score})")
    """

    def __init__(self):
        self._wiki_root: Path | None = None
        self._rag_store = None
        self._skill_root: Path | None = None

    # ------------------------------------------------------------------
    # Cross Search — 统一查询
    # ------------------------------------------------------------------

    def cross_search(
        self,
        query: str,
        top_k: int = 10,
        sources: list[str] | None = None,
    ) -> list[KnowledgeItem]:
        """跨知识库统一查询。

        同时搜索 Wiki、RAG（文献+协议）、Skill，合并结果按相关性排序。

        Args:
            query: 查询文本（中文/英文）
            top_k: 返回的最大结果数
            sources: 限定搜索的知识源（默认全部）

        Returns:
            按 score 降序排列的知识条目列表
        """
        sources = sources or KnowledgeSource.all_sources()
        all_results: list[KnowledgeItem] = []
        labels = KnowledgeSource.labels()

        # 1. 搜索 Wiki
        if KnowledgeSource.WIKI in sources:
            wiki_results = self._search_wiki(query, top_k)
            all_results.extend(wiki_results)

        # 2. 搜索 RAG 文献
        if KnowledgeSource.RAG_LITERATURE in sources:
            lit_results = self._search_rag_literature(query, top_k)
            all_results.extend(lit_results)

        # 3. 搜索 RAG 协议
        if KnowledgeSource.RAG_PROTOCOL in sources:
            prot_results = self._search_rag_protocols(query, top_k)
            all_results.extend(prot_results)

        # 4. 搜索 Skill
        if KnowledgeSource.SKILL in sources:
            skill_results = self._search_skills(query, top_k)
            all_results.extend(skill_results)

        # 去重：基于标题的 Jaccard 相似度
        all_results = self._deduplicate(all_results)

        # 按分数排序
        all_results.sort(key=lambda r: (-r.score, r.source))
        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Cross Enhancement — 交叉增强
    # ------------------------------------------------------------------

    def cross_enhance(
        self,
        wiki_slug: str | None = None,
        skill_slug: str | None = None,
        rag_protocol_id: str | None = None,
    ) -> dict:
        """交叉增强：从一条知识出发，找到其他知识源中的关联内容。

        Args:
            wiki_slug: Wiki 条目 slug
            skill_slug: Skill slug
            rag_protocol_id: RAG 协议 ID

        Returns:
            包含 'source_item', 'related_wiki', 'related_skills',
            'related_protocols', 'related_literature' 的字典
        """
        result: dict[str, Any] = {
            "source_item": None,
            "related_wiki": [],
            "related_skills": [],
            "related_protocols": [],
            "related_literature": [],
        }

        search_text = ""

        # 确定源条目
        if wiki_slug:
            entry = self._get_wiki_entry(wiki_slug)
            if entry:
                result["source_item"] = entry
                search_text = entry.get("title", "") + " " + entry.get("content", "")
        elif skill_slug:
            skill = self._get_skill(skill_slug)
            if skill:
                result["source_item"] = skill
                search_text = skill.get("title", "") + " " + skill.get("content", "")
        elif rag_protocol_id:
            prot = self._get_rag_protocol(rag_protocol_id)
            if prot:
                result["source_item"] = prot
                search_text = prot.get("title", "") + " " + prot.get("summary", "")

        if not search_text:
            return result

        # 用源条目的文本作为查询，搜索其他知识源
        related = self.cross_search(search_text, top_k=15)

        for item in related:
            if item.source == KnowledgeSource.WIKI and item.id != wiki_slug:
                result["related_wiki"].append(item.to_dict())
            elif item.source == KnowledgeSource.SKILL and item.id != skill_slug:
                result["related_skills"].append(item.to_dict())
            elif item.source == KnowledgeSource.RAG_PROTOCOL and item.id != rag_protocol_id:
                result["related_protocols"].append(item.to_dict())
            elif item.source == KnowledgeSource.RAG_LITERATURE:
                result["related_literature"].append(item.to_dict())

        return result

    # ------------------------------------------------------------------
    # Auto-Promotion — Wiki 经验自动提纯为 Skill
    # ------------------------------------------------------------------

    def suggest_skill_from_wiki(
        self,
        min_similarity: float = 0.6,
        min_occurrence: int = 3,
    ) -> list[dict]:
        """检测 Wiki 中重复出现的主题，建议转化为 Skill。

        规则：
        1. 在 Wiki 中找到主题相似度 ≥ min_similarity 的条目组
        2. 条目组大小 ≥ min_occurrence 时触发建议
        3. 返回建议列表，包含合并的内容草稿

        Args:
            min_similarity: 最小 Jaccard 相似度（按词）
            min_occurrence: 最少出现次数阈值

        Returns:
            建议列表，每项包含: suggested_slug, suggested_title,
            source_entries, merged_keywords, confidence
        """
        entries = self._list_wiki_entries()
        if len(entries) < min_occurrence:
            return []

        suggestions: list[dict] = []
        grouped: list[set[int]] = []  # 已分组的条目索引集合

        for i in range(len(entries)):
            if any(i in g for g in grouped):
                continue

            group = [i]
            for j in range(i + 1, len(entries)):
                if any(j in g for g in grouped):
                    continue
                sim = _token_jaccard(
                    entries[i].get("content", ""),
                    entries[j].get("content", ""),
                )
                if sim >= min_similarity:
                    group.append(j)

            if len(group) >= min_occurrence:
                grouped.append(set(group))
                group_entries = [entries[idx] for idx in group]

                # 提取共同关键词
                all_tags: list[str] = []
                for e in group_entries:
                    all_tags.extend(e.get("tags", []))

                # 合并内容摘要
                merged_content = "\n\n---\n\n".join(
                    f"### 来源: {e.get('title', '未命名')}\n{e.get('content', '')[:500]}"
                    for e in group_entries
                )

                from collections import Counter
                tag_counts = Counter(all_tags)
                common_tags = [t for t, c in tag_counts.most_common(5) if c >= 2]

                suggestions.append({
                    "suggested_slug": _slug_from_text(group_entries[0].get("title", "auto-skill")),
                    "suggested_title": f"[自动建议] {group_entries[0].get('title', '未命名')}",
                    "source_entries": [e.get("id", "") for e in group_entries],
                    "occurrence_count": len(group_entries),
                    "merged_keywords": common_tags,
                    "merged_content_preview": merged_content[:2000],
                    "confidence": min(1.0, len(group_entries) / (min_occurrence + 2) + 0.3),
                })

        return suggestions

    # ------------------------------------------------------------------
    # Knowledge Graph Context — 系统提示注入
    # ------------------------------------------------------------------

    def knowledge_graph_context(self, query: str | None = None) -> str:
        """生成跨知识库的知识图谱上下文，用于注入系统提示。

        Args:
            query: 可选的当前查询（用于聚焦相关条目）

        Returns:
            格式化的 Markdown 文本
        """
        lines = ["## 📚 统一知识库 (Unified Knowledge Base)", ""]

        # Wiki 统计
        wiki_count = len(self._list_wiki_entries())
        lines.append(f"- 📝 Wiki 经验条目: **{wiki_count}** 条")

        # Skill 统计
        try:
            from .skills import list_skills
            skill_count = len(list_skills())
            lines.append(f"- 🔧 流程技能: **{skill_count}** 个")
        except Exception:
            lines.append("- 🔧 流程技能: 不可用")

        # RAG 统计
        try:
            from .bio.rag_store import default_rag_store
            store = default_rag_store()
            stats = store.stats()
            lit = stats.get("literature", {})
            prot = stats.get("protocols", {})
            lines.append(f"- 📄 文献记录: **{lit.get('articles', 0)}** 篇")
            lines.append(f"- 🧪 协议记录: **{prot.get('records', 0)}** 条")
        except Exception:
            lines.append("- 📄 文献记录: 不可用")
            lines.append("- 🧪 协议记录: 不可用")

        lines.append("")

        # 如果有查询，搜索相关内容
        if query:
            results = self.cross_search(query, top_k=8)
            if results:
                lines.append(f"### 🔍 与「{query}」相关的知识", "")
                for item in results:
                    lines.append(
                        f"- [{item.source_label}] **{item.title}** "
                        f"(score={item.score:.2f})"
                    )
                    if item.tags:
                        lines.append(f"  标签: {', '.join(item.tags[:5])}")
                lines.append("")

        # 自动技能建议
        suggestions = self.suggest_skill_from_wiki()
        if suggestions:
            lines.append("### 💡 技能转化建议", "")
            for s in suggestions[:3]:
                lines.append(
                    f"- 🎯 **{s['suggested_title']}** "
                    f"(来源: {s['occurrence_count']} 条 Wiki 经验, "
                    f"置信度: {s['confidence']:.0%})"
                )
            lines.append("")

        # 交叉增强提示
        lines.extend([
            "### 🔗 交叉检索提示",
            "",
            "- 使用 `kb_cross_search` 跨库查询相关知识",
            "- 使用 `kb_cross_enhance` 从一条知识出发找关联",
            "- 使用 `kb_suggest_skill` 检查是否有可自动提纯的 Wiki 经验",
            "- Wiki、RAG、Skill 的知识会自动交叉引用，增强回答质量",
            "",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers — Wiki
    # ------------------------------------------------------------------

    def _search_wiki(self, query: str, top_k: int) -> list[KnowledgeItem]:
        """搜索 Wiki 知识库。"""
        try:
            from .wiki import search_entries
            entries = search_entries(query, limit=top_k)
            return [
                KnowledgeItem(
                    source=KnowledgeSource.WIKI,
                    source_label=KnowledgeSource.labels()[KnowledgeSource.WIKI],
                    id=e.get("id", ""),
                    title=e.get("title", ""),
                    content=e.get("content", ""),
                    score=self._compute_score(query, e.get("title", "") + " " + e.get("content", "")),
                    tags=e.get("tags", []),
                    created_at=e.get("created_at", ""),
                )
                for e in entries
            ]
        except Exception:
            return []

    def _list_wiki_entries(self) -> list[dict]:
        """列出所有 Wiki 条目。"""
        try:
            from .wiki import list_entries
            return [
                {
                    "id": e.id if hasattr(e, 'id') else "",
                    "title": e.title if hasattr(e, 'title') else "",
                    "content": e.content if hasattr(e, 'content') else "",
                    "tags": e.tags if hasattr(e, 'tags') else [],
                    "created_at": e.created_at if hasattr(e, 'created_at') else "",
                }
                for e in list_entries()
            ]
        except Exception:
            return []

    def _get_wiki_entry(self, slug: str) -> dict | None:
        """获取单条 Wiki 条目。"""
        try:
            from .wiki import get_entry
            entry = get_entry(slug)
            if entry:
                return {
                    "id": getattr(entry, "id", slug),
                    "title": getattr(entry, "title", ""),
                    "content": getattr(entry, "content", ""),
                    "tags": getattr(entry, "tags", []),
                    "created_at": getattr(entry, "created_at", ""),
                }
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Private helpers — RAG
    # ------------------------------------------------------------------

    def _search_rag_literature(self, query: str, top_k: int) -> list[KnowledgeItem]:
        """搜索 RAG 文献库。"""
        try:
            from .bio.rag_store import default_rag_store
            store = default_rag_store()
            results = store.search_literature(query=query, top_k=top_k)
            return [
                KnowledgeItem(
                    source=KnowledgeSource.RAG_LITERATURE,
                    source_label=KnowledgeSource.labels()[KnowledgeSource.RAG_LITERATURE],
                    id=str(r.get("pmid", r.get("article_id", ""))),
                    title=r.get("title", ""),
                    content=r.get("abstract", r.get("summary", "")),
                    score=r.get("score", 0.0),
                    tags=r.get("methods", []),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{r.get('pmid', '')}" if r.get("pmid") else "",
                    created_at=r.get("pub_date", ""),
                    metadata={"pmid": r.get("pmid", ""), "journal": r.get("journal", "")},
                )
                for r in results
            ]
        except Exception:
            return []

    def _search_rag_protocols(self, query: str, top_k: int) -> list[KnowledgeItem]:
        """搜索 RAG 协议库。"""
        try:
            from .bio.rag_store import default_rag_store
            store = default_rag_store()
            results = store.search_protocols(query=query, top_k=top_k)
            return [
                KnowledgeItem(
                    source=KnowledgeSource.RAG_PROTOCOL,
                    source_label=KnowledgeSource.labels()[KnowledgeSource.RAG_PROTOCOL],
                    id=str(r.get("protocol_id", "")),
                    title=r.get("title", ""),
                    content=r.get("summary", r.get("description", "")),
                    score=r.get("score", 0.0),
                    tags=r.get("tags", []),
                    url=r.get("url", ""),
                    created_at=r.get("collected_at", ""),
                    metadata={"source": r.get("source", ""), "domain": r.get("domain", "")},
                )
                for r in results
            ]
        except Exception:
            return []

    def _get_rag_protocol(self, protocol_id: str) -> dict | None:
        """获取单条 RAG 协议记录。"""
        try:
            from .bio.rag_store import default_rag_store
            store = default_rag_store()
            results = store.search_protocols(query=protocol_id, top_k=1)
            if results:
                r = results[0]
                return {
                    "id": str(r.get("protocol_id", "")),
                    "title": r.get("title", ""),
                    "summary": r.get("summary", ""),
                    "tags": r.get("tags", []),
                    "url": r.get("url", ""),
                }
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Private helpers — Skill
    # ------------------------------------------------------------------

    def _search_skills(self, query: str, top_k: int) -> list[KnowledgeItem]:
        """搜索 Skill 库。"""
        try:
            from .skills import search_skills, read_skill
            metas = search_skills(query, limit=top_k)
            items = []
            for meta in metas:
                try:
                    content = read_skill(meta.slug)[:1000]
                except Exception:
                    content = meta.summary
                items.append(KnowledgeItem(
                    source=KnowledgeSource.SKILL,
                    source_label=KnowledgeSource.labels()[KnowledgeSource.SKILL],
                    id=meta.slug,
                    title=meta.title,
                    content=content,
                    score=self._compute_score(query, meta.title + " " + meta.summary),
                    tags=meta.tags,
                    created_at=meta.updated_at,
                ))
            return items
        except Exception:
            return []

    def _get_skill(self, slug: str) -> dict | None:
        """获取单个 Skill。"""
        try:
            from .skills import read_skill
            content = read_skill(str(slug))
            return {
                "id": slug,
                "title": slug,
                "content": content[:2000],
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_score(self, query: str, text: str) -> float:
        """计算查询与文本的相关性分数。"""
        query_terms = set(re.findall(r"[\w一-鿿]+", query.lower()))
        text_lower = text.lower()
        if not query_terms:
            return 0.0
        matched = sum(1 for t in query_terms if t in text_lower)
        return min(1.0, matched / len(query_terms))

    def _deduplicate(self, items: list[KnowledgeItem], threshold: float = 0.75) -> list[KnowledgeItem]:
        """去重：移除标题相似度 ≥ threshold 的重复条目（保留分数最高的）。"""
        if len(items) <= 1:
            return items

        kept: list[KnowledgeItem] = []
        for item in sorted(items, key=lambda x: -x.score):
            is_dup = False
            for k in kept:
                if _token_jaccard(item.title, k.title) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(item)
        return kept


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_GLOBAL_KB: UnifiedKnowledgeBase | None = None


def get_knowledge_base() -> UnifiedKnowledgeBase:
    """获取全局统一知识库单例。"""
    global _GLOBAL_KB
    if _GLOBAL_KB is None:
        _GLOBAL_KB = UnifiedKnowledgeBase()
    return _GLOBAL_KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_jaccard(text1: str, text2: str) -> float:
    """计算两个文本的 Jaccard 相似度（字符级 3-gram）。"""
    if not text1 or not text2:
        return 0.0

    def ngrams(s: str, n: int = 3) -> set:
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    a = ngrams(text1)
    b = ngrams(text2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _slug_from_text(text: str) -> str:
    """从文本生成 slug。"""
    slug = re.sub(r"[^a-z0-9一-鿿]+", "-", text.lower()).strip("-")
    return slug[:64] if slug else "auto-skill"
