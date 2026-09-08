"""UnifiedKnowledgeBase (C4) 的交叉检索/增强/提纯逻辑测试.

对应简历声明「多源统一知识库」的工程能力。通过 monkeypatch 替换掉
底层懒加载的 wiki / RAG / skill 存储入口，用受控数据隔离出
UnifiedKnowledgeBase 自身的编排逻辑：
  - cross_search     : 混合来源 + 按标题去重 + 按分数排序 + top_k
  - cross_enhance    : 关联结果不含源条目自身
  - suggest_skill_from_wiki : ≥3 相似条目给出建议，低于阈值为空

这些是纯离线可跑的单元测试，不依赖 Postgres / 真实 wiki 文件。
"""

from __future__ import annotations

import dataclasses

from corecoder.knowledge_base import (
    KnowledgeItem,
    UnifiedKnowledgeBase,
    KnowledgeSource,
)


def _item(source: str, id: str, title: str, score: float, content: str = "c") -> KnowledgeItem:
    """构造一个 KnowledgeItem，source_label 用枚举映射。"""
    labels = KnowledgeSource.labels()
    return KnowledgeItem(
        source=source,
        source_label=labels.get(source, source),
        id=id,
        title=title,
        content=content,
        score=score,
    )


# ---------------------------------------------------------------------------
# cross_search — 混合来源 + 去重 + 排序 + top_k
# ---------------------------------------------------------------------------


def test_cross_search_merges_sources_and_sort_by_score(monkeypatch):
    kb = UnifiedKnowledgeBase()

    # 让四个来源各自返回一个不同标题的条目
    monkeypatch.setattr(
        kb, "_search_wiki",
        lambda q, top_k: [_item(KnowledgeSource.WIKI, "w1", "对比 STAR 与 HISAT2 的比对差异", 0.4, "wiki")],
    )
    monkeypatch.setattr(
        kb, "_search_rag_literature",
        lambda q, top_k: [_item(KnowledgeSource.RAG_LITERATURE, "pmid123", "RNA-seq 差异表达流程综述", 0.9, "lit")],
    )
    monkeypatch.setattr(
        kb, "_search_rag_protocols",
        lambda q, top_k: [_item(KnowledgeSource.RAG_PROTOCOL, "p1", "DESeq2 协议标准化步骤", 0.7, "prot")],
    )
    monkeypatch.setattr(
        kb, "_search_skills",
        lambda q, top_k: [_item(KnowledgeSource.SKILL, "s1", "差异表达分析通用流程", 0.6, "skill")],
    )

    results = kb.cross_search("RNA-seq 差异表达 DESeq2", top_k=10)

    # 四个来源都被并入
    assert len(results) == 4
    sources = [r.source for r in results]
    assert KnowledgeSource.WIKI in sources
    assert KnowledgeSource.RAG_LITERATURE in sources
    assert KnowledgeSource.RAG_PROTOCOL in sources
    assert KnowledgeSource.SKILL in sources

    # 按 score 降序
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].source == KnowledgeSource.RAG_LITERATURE  # score 0.9
    assert results[0].score == 0.9


def test_cross_search_dedup_by_title_keeps_highest_score(monkeypatch):
    kb = UnifiedKnowledgeBase()

    # Wiki 与文献条目标题完全相同，但文献分数更高 → 应保留文献、去掉 Wiki。
    monkeypatch.setattr(
        kb, "_search_wiki",
        lambda q, top_k: [_item(KnowledgeSource.WIKI, "w1", "RNA-seq 差异表达标准流程", 0.5, "wiki")],
    )
    monkeypatch.setattr(
        kb, "_search_rag_literature",
        lambda q, top_k: [_item(KnowledgeSource.RAG_LITERATURE, "pmid1", "RNA-seq 差异表达标准流程", 0.9, "lit")],
    )
    monkeypatch.setattr(kb, "_search_rag_protocols", lambda q, top_k: [])
    monkeypatch.setattr(kb, "_search_skills", lambda q, top_k: [])

    results = kb.cross_search("RNA-seq 差异表达", top_k=10)

    # 同标题被去重，仅保留分数更高的文献条目
    assert len(results) == 1
    assert results[0].source == KnowledgeSource.RAG_LITERATURE
    assert results[0].score == 0.9


def test_cross_search_respects_top_k(monkeypatch):
    kb = UnifiedKnowledgeBase()

    monkeypatch.setattr(
        kb, "_search_wiki",
        lambda q, top_k: [
            _item(KnowledgeSource.WIKI, f"w{i}", title, 0.5 + i * 0.01, "c")
            for i, title in enumerate([
                "RNA-seq 比对与定量分析",
                "单细胞测序质控与降维",
                "全基因组变异检测流程",
                "蛋白质结构预测方法",
                "宏基因组物种注释",
            ])
        ],
    )
    monkeypatch.setattr(kb, "_search_rag_literature", lambda q, top_k: [])
    monkeypatch.setattr(kb, "_search_rag_protocols", lambda q, top_k: [])
    monkeypatch.setattr(kb, "_search_skills", lambda q, top_k: [])

    results = kb.cross_search("wiki", top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# cross_enhance — 关联结果不含源条目自身
# ---------------------------------------------------------------------------


def test_cross_enhance_excludes_source_itself(monkeypatch):
    kb = UnifiedKnowledgeBase()

    src = {"id": "my-wiki", "title": "DESeq2 标准化", "content": "如何用 DESeq2 做标准化。"}

    def fake_get_wiki(slug):
        return src if slug == "my-wiki" else None

    def fake_cross_search(query, top_k=10):
        return [
            _item(KnowledgeSource.WIKI, "my-wiki", "DESeq2 标准化", 0.95, "source 自身"),
            _item(KnowledgeSource.WIKI, "other-wiki", "DESeq2 校正", 0.6, "另一条 wiki"),
            _item(KnowledgeSource.SKILL, "some-skill", "DESeq2 技能", 0.7, "skill"),
            _item(KnowledgeSource.RAG_PROTOCOL, "prot-1", "DESeq2 协议", 0.8, "prot"),
            _item(KnowledgeSource.RAG_LITERATURE, "lit-1", "DESeq2 文献", 0.9, "lit"),
        ]

    monkeypatch.setattr(kb, "_get_wiki_entry", fake_get_wiki)
    monkeypatch.setattr(kb, "cross_search", fake_cross_search)

    result = kb.cross_enhance(wiki_slug="my-wiki")

    # 源条目本身被视为 source_item
    assert result["source_item"]["id"] == "my-wiki"

    # related_wiki 不含源条目自身（id 为 "my-wiki" 的那条被过滤）
    related_ids = [i["id"] for i in result["related_wiki"]]
    assert "my-wiki" not in related_ids
    assert "other-wiki" in related_ids

    # 其他来源各自收集
    assert [i["id"] for i in result["related_skills"]] == ["some-skill"]
    assert [i["id"] for i in result["related_protocols"]] == ["prot-1"]
    assert [i["id"] for i in result["related_literature"]] == ["lit-1"]


def test_cross_enhance_empty_when_no_source(monkeypatch):
    kb = UnifiedKnowledgeBase()
    monkeypatch.setattr(kb, "cross_search", lambda q, top_k=10: [])
    result = kb.cross_enhance(wiki_slug="missing-slug")
    assert result["source_item"] is None
    assert result["related_wiki"] == []
    assert result["related_skills"] == []


# ---------------------------------------------------------------------------
# suggest_skill_from_wiki — ≥3 相似条目触发建议
# ---------------------------------------------------------------------------


def _wiki_entry(id_: str, title: str, content: str, tags: list[str]) -> dict:
    return {
        "id": id_,
        "title": title,
        "content": content,
        "tags": tags,
        "created_at": "",
    }


SHARED_CONTENT = (
    "执行 RNA-seq 差异表达时先用 STAR 比对到参考基因组，再合并样本计数矩阵，"
    "并用 DESeq2 做标准化与差异显著性检验，最后对显著基因做通路富集分析。"
)


def test_suggest_skill_from_wiki_when_three_similar(monkeypatch):
    kb = UnifiedKnowledgeBase()
    entries = [
        _wiki_entry("w1", "DESeq2 流程一", SHARED_CONTENT, ["rnaseq", "deseq2"]),
        _wiki_entry("w2", "DESeq2 流程二", SHARED_CONTENT, ["rnaseq", "deseq2"]),
        _wiki_entry("w3", "DESeq2 流程三", SHARED_CONTENT, ["rnaseq", "deseq2"]),
    ]
    monkeypatch.setattr(kb, "_list_wiki_entries", lambda: entries)

    suggestions = kb.suggest_skill_from_wiki(min_similarity=0.6, min_occurrence=3)
    assert len(suggestions) == 1

    s = suggestions[0]
    assert s["occurrence_count"] == 3
    assert set(s["source_entries"]) == {"w1", "w2", "w3"}
    # 共同关键词来自出现 ≥2 次的标签
    assert "deseq2" in s["merged_keywords"]
    assert "rnaseq" in s["merged_keywords"]
    assert s["confidence"] > 0.0


def test_suggest_skill_from_wiki_empty_below_threshold(monkeypatch):
    kb = UnifiedKnowledgeBase()
    entries = [
        _wiki_entry("w1", "差异表达流程", SHARED_CONTENT, ["rnaseq"]),
        _wiki_entry("w2", "另一个主题", "完全无关的测序质量评估内容。", ["qc"]),
    ]
    monkeypatch.setattr(kb, "_list_wiki_entries", lambda: entries)

    suggestions = kb.suggest_skill_from_wiki(min_similarity=0.6, min_occurrence=3)
    assert suggestions == []


def test_suggest_skill_from_wiki_respects_min_occurrence(monkeypatch):
    """两个相似条目低于 min_occurrence=3 时不给建议。"""
    kb = UnifiedKnowledgeBase()
    entries = [
        _wiki_entry("w1", "DESeq2 流程一", SHARED_CONTENT, ["rnaseq"]),
        _wiki_entry("w2", "DESeq2 流程二", SHARED_CONTENT, ["rnaseq"]),
    ]
    monkeypatch.setattr(kb, "_list_wiki_entries", lambda: entries)

    assert kb.suggest_skill_from_wiki(min_similarity=0.6, min_occurrence=3) == []
