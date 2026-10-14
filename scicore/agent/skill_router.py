"""Skill 路由与分层机制 — SciCoreAgent 技能调度核心.

实现三层技能体系：
  - 核心层 (core)：基础生信流程（RNA-seq pipeline、protocol extraction 等）
  - 领域层 (domain)：领域专项技能（cell biology、genomics 等）
  - 辅助层 (auxiliary)：辅助工具技能（格式转换、可视化等）

功能：
  1. 意图 → 技能映射：基于关键词 + LLM 语义匹配
  2. 技能链式调用：前驱技能的输出自动成为后继技能的输入
  3. 分层权限：不同层级拥有不同的工具白名单
  4. 前置条件检查：确保依赖技能已执行
  5. 互斥检测：避免冲突技能同时执行

Reference: SciCoreAgent 需求文档, Section 5 (Skill 路由).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Skill Tier — 三层技能分层
# ---------------------------------------------------------------------------


class SkillTier(str, Enum):
    """技能层级。高层级技能可以调用低层级技能，反之不行。"""

    CORE = "core"  # 核心层：基础生信流程，工具白名单最广
    DOMAIN = "domain"  # 领域层：领域专项，工具白名单受限
    AUXILIARY = "auxiliary"  # 辅助层：辅助工具，工具白名单最窄


# ---------------------------------------------------------------------------
# Skill Route — 单条技能路由条目
# ---------------------------------------------------------------------------


@dataclass
class SkillRoute:
    """一条已注册的技能路由条目。

    Attributes:
        slug: 技能唯一标识（与 skills.py 中的 slug 对应）
        title: 中文标题
        tier: 技能层级
        priority: 优先级（0-100，越高越优先匹配）
        triggers: 触发关键词/短语列表（支持中英文）
        preconditions: 前置技能 slug 列表（必须先执行）
        mutual_exclusion: 互斥技能 slug 列表（不能同时执行）
        next_skills: 可链式调用的后继技能 slug 列表
        required_tools: 该技能需要的工具名称列表
        input_schema: 输入参数的 JSON Schema（可选）
        output_schema: 输出参数的 JSON Schema（可选）
        auto_chain: 是否自动触发后继技能
    """

    slug: str
    title: str
    tier: SkillTier = SkillTier.DOMAIN
    priority: int = 50
    triggers: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    mutual_exclusion: list[str] = field(default_factory=list)
    next_skills: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict | None = None
    output_schema: dict | None = None
    auto_chain: bool = False
    tags: list[str] = field(default_factory=list)
    applicable_when: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "tier": self.tier.value,
            "priority": self.priority,
            "triggers": self.triggers,
            "preconditions": self.preconditions,
            "mutual_exclusion": self.mutual_exclusion,
            "next_skills": self.next_skills,
            "required_tools": self.required_tools,
            "auto_chain": self.auto_chain,
            "tags": self.tags,
            "applicable_when": self.applicable_when,
            "boundaries": self.boundaries,
            "examples": self.examples,
        }


# ---------------------------------------------------------------------------
# Routing Result
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    """路由匹配结果。"""

    skill: SkillRoute
    score: float  # 0.0 - 1.0 匹配置信度
    matched_triggers: list[str]  # 命中的触发词
    missing_preconditions: list[str]  # 缺失的前置技能
    conflicts: list[str]  # 已激活的冲突技能
    next_available: list[str]  # 可链式调用的后继技能

    @property
    def is_ready(self) -> bool:
        """是否可以直接执行（无缺失前置、无冲突）。"""
        return len(self.missing_preconditions) == 0 and len(self.conflicts) == 0


# ---------------------------------------------------------------------------
# Skill Router
# ---------------------------------------------------------------------------


class SkillRouter:
    """技能路由器 — 意图匹配、分层调度、链式编排。

    Usage::

        router = SkillRouter()
        router.register(SkillRoute(
            slug="rnaseq-pipeline",
            title="RNA-seq 分析流程",
            tier=SkillTier.CORE,
            triggers=["RNA-seq", "转录组", "差异表达", "differential expression"],
            next_skills=["protocol-extraction"],
        ))
        results = router.route("我想做RNA-seq差异表达分析")
    """

    def __init__(self):
        self._routes: dict[str, SkillRoute] = {}
        self._active_skills: set[str] = set()  # 当前会话已执行的技能
        self._chain_history: list[str] = []  # 技能执行链
        self.last_route_diagnostics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, route: SkillRoute) -> None:
        """注册一条技能路由。"""
        if route.slug in self._routes:
            raise ValueError(f"技能 '{route.slug}' 已注册，使用 update() 更新")
        self._validate_route(route)
        self._routes[route.slug] = route

    def update(self, slug: str, **kwargs) -> None:
        """更新已注册的技能路由。"""
        if slug not in self._routes:
            raise KeyError(f"技能 '{slug}' 未注册")
        route = self._routes[slug]
        for key, value in kwargs.items():
            if hasattr(route, key):
                setattr(route, key, value)
        self._validate_route(route)

    def unregister(self, slug: str) -> None:
        """注销一条技能路由。"""
        self._routes.pop(slug, None)
        self._active_skills.discard(slug)

    def get(self, slug: str) -> SkillRoute | None:
        """按 slug 获取路由条目。"""
        return self._routes.get(slug)

    def list_all(self) -> list[SkillRoute]:
        """列出所有已注册的技能路由。"""
        return sorted(self._routes.values(), key=lambda r: (-r.priority, r.slug))

    def list_by_tier(self, tier: SkillTier) -> list[SkillRoute]:
        """按层级列出技能。"""
        return [r for r in self._routes.values() if r.tier == tier]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        user_intent: str,
        max_results: int = 5,
        min_score: float = 0.1,
        candidate_limit: int = 30,
    ) -> list[RoutingResult]:
        """根据用户意图匹配最合适的技能。

        二阶段匹配策略：先从紧凑 Skill Catalog 元数据召回有界候选集，
        再结合触发词、适用条件、示例、边界、层级和优先级进行精排。

        精排信号：
        1. 精确关键词匹配（权重 0.8）
        2. 部分关键词匹配（权重 0.5）
        3. 语义关联匹配（权重 0.3）
        4. 已执行过的技能降权（×0.7）
        """
        intent_lower = user_intent.lower()
        results: list[RoutingResult] = []
        candidates = self._recall_candidates(user_intent, limit=candidate_limit)

        for route, recall_score, _recall_reasons in candidates:
            if any(
                boundary.strip().lower() in intent_lower
                for boundary in route.boundaries
                if boundary.strip()
            ):
                continue
            matched_triggers: list[str] = []
            score = 0.0

            for trigger in route.triggers:
                trigger_lower = trigger.lower()
                if trigger_lower in intent_lower:
                    # 精确匹配
                    matched_triggers.append(trigger)
                    score += 0.8
                elif any(
                    word in intent_lower
                    for word in trigger_lower.split()
                    if len(word) >= 3
                ):
                    # 部分匹配
                    matched_triggers.append(trigger)
                    score += 0.5
                elif _fuzzy_match(trigger_lower, intent_lower):
                    # 模糊匹配
                    matched_triggers.append(trigger)
                    score += 0.3

            metadata_matches = [
                value
                for value in (
                    route.tags + route.applicable_when + route.examples + [route.title]
                )
                if _catalog_match(str(value), intent_lower)
            ]
            if not matched_triggers and not metadata_matches:
                continue

            # 归一化分数
            max_possible = len(route.triggers) * 0.8
            trigger_score = min(score / max(max_possible, 0.1), 1.0)
            metadata_score = min(len(metadata_matches) / 3.0, 1.0)
            priority_score = max(0.0, min(float(route.priority) / 100.0, 1.0))
            score = (
                0.62 * trigger_score
                + 0.20 * metadata_score
                + 0.10 * recall_score
                + 0.08 * priority_score
            )

            # 已执行的技能降权
            if route.slug in self._active_skills:
                score *= 0.7

            # 优先给高层级技能加权
            if route.tier == SkillTier.CORE:
                score *= 1.1
            elif route.tier == SkillTier.AUXILIARY:
                score *= 0.9

            score = min(score, 1.0)

            if score < min_score:
                continue

            # 检查前置条件
            missing_preconditions = [
                p for p in route.preconditions if p not in self._active_skills
            ]

            # 检查冲突
            conflicts = [
                c for c in route.mutual_exclusion if c in self._active_skills
            ]

            # 后继技能
            next_available = [
                n for n in route.next_skills if n in self._routes
            ]

            results.append(
                RoutingResult(
                    skill=route,
                    score=round(score, 4),
                    matched_triggers=matched_triggers,
                    missing_preconditions=missing_preconditions,
                    conflicts=conflicts,
                    next_available=next_available,
                )
            )

        # 按分数降序排列
        results.sort(key=lambda r: (-r.score, -r.skill.priority))
        self.last_route_diagnostics = {
            "strategy": "catalog_recall_then_metadata_rerank",
            "catalog_size": len(self._routes),
            "candidate_count": len(candidates),
            "candidate_limit": max(1, int(candidate_limit)),
            "returned_count": min(len(results), max(0, int(max_results))),
            "candidates": [
                {
                    "slug": route.slug,
                    "recall_score": round(recall_score, 4),
                    "reasons": reasons,
                }
                for route, recall_score, reasons in candidates
            ],
        }
        return results[:max_results]

    def _recall_candidates(self, user_intent: str, *, limit: int = 30):
        """Stage 1: retrieve from catalog metadata without loading Skill bodies."""
        intent = str(user_intent).lower()
        candidates = []
        for route in self._routes.values():
            fields = {
                "title": [route.title],
                "trigger": route.triggers,
                "tag": route.tags,
                "applicable_when": route.applicable_when,
                "example": route.examples,
            }
            reasons = []
            score = 0.0
            for field_name, values in fields.items():
                for value in values:
                    value = str(value).strip().lower()
                    if not value:
                        continue
                    if value in intent:
                        reasons.append(f"{field_name}:exact")
                        score += 1.0 if field_name == "trigger" else 0.7
                    elif _catalog_match(value, intent):
                        reasons.append(f"{field_name}:token")
                        score += 0.55 if field_name == "trigger" else 0.35
            if reasons:
                normalized = min(score / max(1.0, len(route.triggers)), 1.0)
                candidates.append((route, normalized, sorted(set(reasons))))
        candidates.sort(key=lambda item: (-item[1], -item[0].priority, item[0].slug))
        return candidates[: max(1, int(limit))]

    def route_exact(self, slug: str) -> RoutingResult | None:
        """按 slug 精确查找路由。"""
        route = self._routes.get(slug)
        if route is None:
            return None
        return RoutingResult(
            skill=route,
            score=1.0,
            matched_triggers=[],
            missing_preconditions=[
                p for p in route.preconditions if p not in self._active_skills
            ],
            conflicts=[
                c for c in route.mutual_exclusion if c in self._active_skills
            ],
            next_available=[
                n for n in route.next_skills if n in self._routes
            ],
        )

    # ------------------------------------------------------------------
    # Chain
    # ------------------------------------------------------------------

    def get_chain(self, start_slug: str, depth: int = 5) -> list[str]:
        """从起始技能获取技能链（BFS 遍历后继技能）。"""
        chain: list[str] = [start_slug]
        visited: set[str] = {start_slug}
        queue: list[str] = [start_slug]

        while queue and len(chain) < depth:
            current = queue.pop(0)
            route = self._routes.get(current)
            if route is None:
                continue
            for next_slug in route.next_skills:
                if next_slug not in visited and next_slug in self._routes:
                    visited.add(next_slug)
                    chain.append(next_slug)
                    queue.append(next_slug)

        return chain

    def get_next_skills(self, slug: str) -> list[SkillRoute]:
        """获取某个技能的后继技能路由。"""
        route = self._routes.get(slug)
        if route is None:
            return []
        return [
            self._routes[n]
            for n in route.next_skills
            if n in self._routes
        ]

    # ------------------------------------------------------------------
    # Active tracking
    # ------------------------------------------------------------------

    def mark_executed(self, slug: str) -> None:
        """标记技能已执行。"""
        self._active_skills.add(slug)
        self._chain_history.append(slug)

    def mark_all_executed(self, slugs: list[str]) -> None:
        """批量标记技能已执行。"""
        for slug in slugs:
            self.mark_executed(slug)

    def is_executed(self, slug: str) -> bool:
        """检查技能是否已执行。"""
        return slug in self._active_skills

    def reset_session(self) -> None:
        """重置会话状态（新对话开始时调用）。"""
        self._active_skills.clear()
        self._chain_history.clear()

    def get_execution_chain(self) -> list[str]:
        """获取当前会话的技能执行链。"""
        return list(self._chain_history)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_route(self, route: SkillRoute) -> None:
        """校验路由条目的引用完整性。"""
        # 检查前置技能是否存在（延迟检查，允许前向引用）
        import logging as _logging
        for pre in route.preconditions:
            if pre not in self._routes:
                _logging.getLogger(__name__).warning(
                    "技能 '%s' 的前置技能 '%s' 尚未注册（将在链式调用时检查）",
                    route.slug, pre
                )

        # 检查互斥技能是否存在（延迟检查，允许前向引用）
        for excl in route.mutual_exclusion:
            if excl not in self._routes:
                _logging.getLogger(__name__).warning(
                    "技能 '%s' 的互斥技能 '%s' 尚未注册（将在链式调用时检查）",
                    route.slug, excl
                )

        # 检查后继技能是否存在（延迟到链式调用时再检查，允许前向引用）
        for next_slug in route.next_skills:
            if next_slug not in self._routes:
                import logging
                logging.getLogger(__name__).warning(
                    "技能 '%s' 的后继技能 '%s' 尚未注册（将在链式调用时检查）",
                    route.slug, next_slug
                )

        # 不能与自己互斥
        if route.slug in route.mutual_exclusion:
            raise ValueError(f"技能 '{route.slug}' 不能与自身互斥")

        # 不能以自己为前置
        if route.slug in route.preconditions:
            raise ValueError(f"技能 '{route.slug}' 不能以自身为前置")

    # ------------------------------------------------------------------
    # 工具白名单 — 按层级裁剪
    # ------------------------------------------------------------------

    @staticmethod
    def tools_for_tier(tier: SkillTier, all_tools: list[Any]) -> list[Any]:
        """根据技能层级返回工具白名单。

        - core: 所有 bio 工具 + read/write/bash
        - domain: bio 工具 + read（不能写）
        - auxiliary: read-only 工具
        """
        _TIER_ALLOWED: dict[SkillTier, set[str]] = {
            SkillTier.CORE: {
                "read_file", "write_file", "edit_file", "bash",
                "grep", "glob",
                "bio_seq_inspect", "bio_sample_sheet_inspect",
                "bio_count_matrix_inspect", "bio_rnaseq_compare",
                "bio_report", "bio_workflow_sketch", "file_hash",
                "bio_ingest_protocol", "bio_extract_protocol",
                "bio_query_evidence", "bio_replication_plan",
                "wiki_save", "wiki_search", "wiki_list",
                "skill_save", "skill_search", "skill_read",
            },
            SkillTier.DOMAIN: {
                "read_file", "grep", "glob",
                "bio_seq_inspect", "bio_sample_sheet_inspect",
                "bio_count_matrix_inspect", "bio_rnaseq_compare",
                "bio_report", "bio_workflow_sketch", "file_hash",
                "bio_ingest_protocol", "bio_extract_protocol",
                "bio_query_evidence", "bio_replication_plan",
                "wiki_search", "wiki_list",
                "skill_search", "skill_read",
            },
            SkillTier.AUXILIARY: {
                "read_file", "grep", "glob",
                "file_hash",
                "wiki_search", "wiki_list",
                "skill_search", "skill_read",
            },
        }
        allowed = _TIER_ALLOWED.get(tier, _TIER_ALLOWED[SkillTier.AUXILIARY])
        return [t for t in all_tools if t.name in allowed]


# ---------------------------------------------------------------------------
# 默认技能路由注册表
# ---------------------------------------------------------------------------


def _register_default_routes(router: SkillRouter) -> None:
    """注册 SciCoreAgent 内置的技能路由。"""

    # ---- Core 层 ----

    router.register(SkillRoute(
        slug="rnaseq-pipeline",
        title="RNA-seq 分析流程",
        tier=SkillTier.CORE,
        priority=95,
        triggers=[
            "RNA-seq", "RNAseq", "转录组", "transcriptome",
            "差异表达", "differential expression", "DEG",
            "基因表达", "gene expression", "mRNA-seq",
            "单细胞转录组", "scRNA-seq", "single cell RNA",
        ],
        next_skills=["literature-review", "protocol-extraction"],
        required_tools=[
            "bio_seq_inspect", "bio_sample_sheet_inspect",
            "bio_count_matrix_inspect", "bio_rnaseq_compare",
            "bio_workflow_sketch", "bio_report",
        ],
        auto_chain=True,
    ))

    router.register(SkillRoute(
        slug="literature-review",
        title="文献调研",
        tier=SkillTier.CORE,
        priority=90,
        triggers=[
            "文献", "文献调研", "literature", "PubMed",
            "论文", "综述", "review", "检索", "search",
            "published", "publication",
        ],
        next_skills=["rnaseq-pipeline", "replication-planning"],
        required_tools=["pubmed_search", "pubmed_fetch_details",
                        "pubmed_extract_rnaseq_methods", "pubmed_save_to_rag"],
        auto_chain=False,
    ))

    router.register(SkillRoute(
        slug="replication-planning",
        title="实验复现规划",
        tier=SkillTier.CORE,
        priority=85,
        triggers=[
            "复现", "replicate", "reproduce", "重复",
            "复制实验", "replication", "reproducibility",
        ],
        preconditions=["protocol-extraction"],
        next_skills=["workflow-generation"],
        required_tools=[
            "bio_ingest_protocol", "bio_extract_protocol",
            "bio_query_evidence", "bio_replication_plan",
        ],
        auto_chain=True,
    ))

    router.register(SkillRoute(
        slug="workflow-generation",
        title="生信工作流生成",
        tier=SkillTier.CORE,
        priority=80,
        triggers=[
            "工作流", "workflow", "pipeline", "Snakemake",
            "Nextflow", "流程", "脚本生成", "code generation",
        ],
        preconditions=["replication-planning"],
        required_tools=["bio_workflow_sketch", "write_file"],
        auto_chain=False,
    ))

    # ---- Domain 层 ----

    router.register(SkillRoute(
        slug="protocol-extraction",
        title="实验协议提取",
        tier=SkillTier.DOMAIN,
        priority=75,
        triggers=[
            "协议", "protocol", "protocols.io", "实验步骤",
            "方法", "methods", "材料", "materials",
            "protocol extraction",
        ],
        preconditions=["literature-review"],
        next_skills=["replication-planning"],
        required_tools=[
            "bio_ingest_protocol", "bio_extract_protocol",
            "bio_query_evidence",
        ],
        auto_chain=True,
    ))

    router.register(SkillRoute(
        slug="cell-biology",
        title="细胞生物学实验",
        tier=SkillTier.DOMAIN,
        priority=70,
        triggers=[
            "细胞", "cell", "培养", "culture", "免疫荧光",
            "immunofluorescence", "IF", "IHC", "免疫组化",
            "western blot", "flow cytometry", "流式",
            "细胞生物学", "cell biology",
        ],
        next_skills=["protocol-extraction"],
        required_tools=[
            "bio_ingest_protocol", "bio_extract_protocol",
            "bio_query_evidence",
        ],
        auto_chain=False,
    ))

    router.register(SkillRoute(
        slug="genomics",
        title="基因组学分析",
        tier=SkillTier.DOMAIN,
        priority=70,
        triggers=[
            "基因组", "genomics", "WGS", "WES", "全基因组",
            "全外显子", "变异检测", "variant calling",
            "GWAS", "genome-wide",
        ],
        next_skills=["literature-review"],
        required_tools=[
            "bio_seq_inspect", "bio_workflow_sketch", "bio_report",
        ],
        auto_chain=False,
    ))

    router.register(SkillRoute(
        slug="chip-seq",
        title="ChIP-seq 分析",
        tier=SkillTier.DOMAIN,
        priority=65,
        triggers=[
            "ChIP-seq", "ChIP", "chip-seq", "染色质免疫沉淀",
            "peak calling", "peak", "组蛋白修饰", "histone",
            "转录因子结合", "transcription factor binding",
        ],
        next_skills=["literature-review"],
        required_tools=[
            "bio_seq_inspect", "bio_workflow_sketch", "bio_report",
        ],
        auto_chain=False,
    ))

    # ---- Auxiliary 层 ----

    router.register(SkillRoute(
        slug="file-inspection",
        title="生信文件检查",
        tier=SkillTier.AUXILIARY,
        priority=60,
        triggers=[
            "检查文件", "查看文件", "inspect", "文件格式",
            "FASTA", "FASTQ", "BAM", "SAM", "VCF", "GTF", "GFF",
            "质量检查", "QC", "fastqc",
        ],
        required_tools=[
            "bio_seq_inspect", "bio_sample_sheet_inspect",
            "bio_count_matrix_inspect", "file_hash",
        ],
        auto_chain=False,
    ))

    router.register(SkillRoute(
        slug="data-visualization",
        title="数据可视化",
        tier=SkillTier.AUXILIARY,
        priority=55,
        triggers=[
            "可视化", "画图", "图表", "visualization", "plot",
            "PCA", "heatmap", "热图", "火山图", "volcano",
            "MA plot", "箱线图", "boxplot",
        ],
        required_tools=["bio_report", "write_file"],
        auto_chain=False,
    ))

    router.register(SkillRoute(
        slug="format-conversion",
        title="格式转换",
        tier=SkillTier.AUXILIARY,
        priority=50,
        triggers=[
            "格式转换", "convert", "转换", "SAM to BAM",
            "GTF to BED", "文件转换",
        ],
        required_tools=["bash", "read_file", "write_file"],
        auto_chain=False,
    ))


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_GLOBAL_ROUTER: SkillRouter | None = None


def get_skill_router() -> SkillRouter:
    """获取全局技能路由器单例。"""
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        _GLOBAL_ROUTER = SkillRouter()
        _register_default_routes(_GLOBAL_ROUTER)
    return _GLOBAL_ROUTER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(pattern: str, text: str, threshold: float = 0.6) -> bool:
    """简单的模糊匹配：检查 pattern 中的字符是否按序出现在 text 中。"""
    if len(pattern) < 3:
        return False
    pi = 0
    for ch in text:
        if ch == pattern[pi]:
            pi += 1
            if pi == len(pattern):
                return True
    return pi / len(pattern) >= threshold


def _catalog_match(pattern: str, text: str) -> bool:
    """Cheap multilingual catalog recall used before metadata reranking."""
    pattern = str(pattern).lower()
    text = str(text).lower()
    if pattern in text:
        return True
    pattern_tokens = set(
        re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", pattern)
    )
    text_tokens = set(
        re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    )
    if pattern_tokens & text_tokens:
        return True
    return _fuzzy_match(pattern, text)


def skill_router_prompt_context(router: SkillRouter | None = None) -> str:
    """生成 Skill 路由的系统提示上下文，注入到 Agent system prompt 中。"""
    if router is None:
        router = get_skill_router()

    routes = router.list_all()
    if not routes:
        return "没有已注册的技能路由。"

    lines = [
        "## 技能路由表 (Skill Router)",
        "",
        "以下技能按分层组织。当用户提出任务时，优先匹配核心层技能，",
        "然后沿技能链自动推荐下一步。",
        "",
        "| 层级 | 技能 | 触发条件 | 前置条件 | 后继技能 |",
        "|------|------|----------|----------|----------|",
    ]

    tier_labels = {SkillTier.CORE: "🔴 核心", SkillTier.DOMAIN: "🟡 领域", SkillTier.AUXILIARY: "🟢 辅助"}

    for route in routes:
        tier_label = tier_labels.get(route.tier, route.tier.value)
        triggers_short = ", ".join(route.triggers[:3])
        if len(route.triggers) > 3:
            triggers_short += f" (+{len(route.triggers) - 3})"
        pre_short = ", ".join(route.preconditions) if route.preconditions else "无"
        next_short = ", ".join(route.next_skills) if route.next_skills else "无"
        lines.append(
            f"| {tier_label} | {route.title} | {triggers_short} | {pre_short} | {next_short} |"
        )

    lines.extend([
        "",
        "**路由规则**：",
        "- 匹配到的核心层技能优先执行",
        "- 前置技能缺失时，自动先执行前置技能",
        "- 勾选 `auto_chain` 的技能完成后，自动推荐后继技能",
        "- 互斥技能不能在同一会话中同时执行",
        "",
    ])

    return "\n".join(lines)
