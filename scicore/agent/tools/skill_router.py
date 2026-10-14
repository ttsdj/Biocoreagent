"""Skill 路由工具 — 将 SkillRouter 暴露为 Agent 可调用的工具。"""

from __future__ import annotations

import json

from .base import Tool
from ..skill_router import get_skill_router, SkillTier


class SkillRouteTool(Tool):
    """根据用户意图匹配最佳技能的工具。

    调用时机：当用户描述一个生信/生物研究任务时，先用此工具查找匹配的技能。
    """

    name = "skill_route"
    description = (
        "根据用户意图匹配最合适的技能。返回匹配的技能列表，包含置信度分数、"
        "缺失的前置条件、冲突检测和可链式调用的后继技能。"
        "使用此工具来确定应该执行哪个技能来处理用户的请求。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "用户的意图描述（中文或英文），例如 '我想做RNA-seq差异表达分析'",
            },
            "max_results": {
                "type": "integer",
                "description": "最多返回多少个匹配结果",
                "default": 5,
            },
        },
        "required": ["intent"],
    }

    def execute(self, intent: str, max_results: int = 5) -> str:
        router = get_skill_router()
        results = router.route(intent, max_results=max_results)

        if not results:
            return json.dumps({
                "matched": False,
                "message": "未找到匹配的技能。请尝试更具体的描述，或使用 skill_route_list 查看所有可用技能。",
                "results": [],
            }, ensure_ascii=False, indent=2)

        output = {
            "matched": True,
            "intent": intent,
            "results": [
                {
                    "slug": r.skill.slug,
                    "title": r.skill.title,
                    "tier": r.skill.tier.value,
                    "score": r.score,
                    "matched_triggers": r.matched_triggers,
                    "is_ready": r.is_ready,
                    "missing_preconditions": r.missing_preconditions,
                    "conflicts": r.conflicts,
                    "next_available": r.next_available,
                    "auto_chain": r.skill.auto_chain,
                }
                for r in results
            ],
        }
        return json.dumps(output, ensure_ascii=False, indent=2)


class SkillChainTool(Tool):
    """查询技能链的工具。

    给定一个技能 slug，返回从该技能开始的完整技能链。
    """

    name = "skill_chain"
    description = (
        "给定一个技能 slug，返回从该技能开始可触发的完整技能链。"
        "用于规划多步骤任务时了解技能之间的依赖关系。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "起始技能的 slug，例如 'rnaseq-pipeline'",
            },
            "depth": {
                "type": "integer",
                "description": "链的最大深度（默认 5）",
                "default": 5,
            },
        },
        "required": ["slug"],
    }

    def execute(self, slug: str, depth: int = 5) -> str:
        router = get_skill_router()
        route = router.get(slug)

        if route is None:
            return json.dumps({
                "found": False,
                "message": f"技能 '{slug}' 未注册。使用 skill_route_list 查看所有可用技能。",
            }, ensure_ascii=False, indent=2)

        chain = router.get_chain(slug, depth=depth)
        chain_details = []
        for s in chain:
            r = router.get(s)
            if r:
                chain_details.append({
                    "slug": r.slug,
                    "title": r.title,
                    "tier": r.tier.value,
                    "preconditions": r.preconditions,
                    "next_skills": r.next_skills,
                    "auto_chain": r.auto_chain,
                })

        return json.dumps({
            "found": True,
            "start_slug": slug,
            "chain": chain,
            "chain_length": len(chain),
            "details": chain_details,
        }, ensure_ascii=False, indent=2)


class SkillRouteListTool(Tool):
    """列出所有已注册技能路由的工具。"""

    name = "skill_route_list"
    description = (
        "列出所有已注册的技能路由，可按层级过滤。"
        "用于了解系统中有哪些可用技能及其分层归属。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "description": "按层级过滤：core（核心层）、domain（领域层）、auxiliary（辅助层）。不提供则列出全部。",
                "enum": ["core", "domain", "auxiliary"],
            },
        },
        "required": [],
    }

    def execute(self, tier: str | None = None) -> str:
        router = get_skill_router()

        if tier:
            try:
                tier_enum = SkillTier(tier)
            except ValueError:
                return json.dumps({
                    "error": f"无效的层级 '{tier}'。有效值: core, domain, auxiliary",
                }, ensure_ascii=False, indent=2)
            routes = router.list_by_tier(tier_enum)
        else:
            routes = router.list_all()

        tier_labels = {
            SkillTier.CORE: "核心层",
            SkillTier.DOMAIN: "领域层",
            SkillTier.AUXILIARY: "辅助层",
        }

        return json.dumps({
            "total": len(routes),
            "skills": [
                {
                    "slug": r.slug,
                    "title": r.title,
                    "tier": r.tier.value,
                    "tier_label": tier_labels.get(r.tier, r.tier.value),
                    "priority": r.priority,
                    "triggers": r.triggers,
                    "preconditions": r.preconditions,
                    "next_skills": r.next_skills,
                    "required_tools": r.required_tools,
                    "auto_chain": r.auto_chain,
                }
                for r in routes
            ],
        }, ensure_ascii=False, indent=2)


class SkillMarkExecutedTool(Tool):
    """标记技能已执行的工具。"""

    name = "skill_mark_executed"
    description = (
        "标记一个或多个技能为已执行。在技能成功完成后调用此工具，"
        "以便路由系统正确追踪执行历史和处理技能链。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "slugs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已执行的技能 slug 列表",
            },
        },
        "required": ["slugs"],
    }

    def execute(self, slugs: list[str]) -> str:
        router = get_skill_router()
        valid_slugs = []
        invalid_slugs = []

        for slug in slugs:
            if router.get(slug):
                valid_slugs.append(slug)
            else:
                invalid_slugs.append(slug)

        router.mark_all_executed(valid_slugs)

        # 自动推荐后继技能
        next_recommendations = []
        for slug in valid_slugs:
            next_skills = router.get_next_skills(slug)
            for ns in next_skills:
                if ns.auto_chain and not router.is_executed(ns.slug):
                    next_recommendations.append({
                        "slug": ns.slug,
                        "title": ns.title,
                        "reason": f"技能 '{slug}' 的自动后继",
                    })

        return json.dumps({
            "marked_executed": valid_slugs,
            "invalid_slugs": invalid_slugs,
            "execution_chain": router.get_execution_chain(),
            "next_recommendations": next_recommendations,
        }, ensure_ascii=False, indent=2)
