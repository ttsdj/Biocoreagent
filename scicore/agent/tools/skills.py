"""Tools for persistent skill sedimentation."""

from __future__ import annotations

import json

from .base import Tool
from ..skills import install_skill_from_path, install_skill_from_url, list_skills, read_skill, save_skill, search_skills


class SkillSaveTool(Tool):
    name = "skill_save"
    description = "Persist a reusable task procedure as a local SciCoreAgent skill Markdown file."
    parameters = {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Stable lowercase id, e.g. rnaseq-count-qc"},
            "title": {"type": "string", "description": "Human-readable skill title"},
            "summary": {"type": "string", "description": "One-sentence summary"},
            "trigger": {"type": "string", "description": "When the agent should reuse this skill"},
            "procedure": {"type": "string", "description": "Reusable step-by-step procedure"},
            "pitfalls": {"type": "string", "description": "Known risks, failure modes, or checks"},
            "interview_questions": {"type": "string", "description": "Likely interview questions and answer cues"},
            "source_task": {"type": "string", "description": "Original user task or project context"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Search tags"},
            "overwrite": {"type": "boolean", "description": "Overwrite an existing skill. Default false."},
        },
        "required": ["slug", "title", "summary", "trigger", "procedure"],
    }

    def execute(
        self,
        slug: str,
        title: str,
        summary: str,
        trigger: str,
        procedure: str,
        pitfalls: str = "",
        interview_questions: str = "",
        source_task: str = "",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> str:
        try:
            meta = save_skill(
                slug=slug,
                title=title,
                summary=summary,
                trigger=trigger,
                procedure=procedure,
                pitfalls=pitfalls,
                interview_questions=interview_questions,
                source_task=source_task,
                tags=tags or [],
                overwrite=overwrite,
            )
            return json.dumps(meta.__dict__, indent=2)
        except Exception as e:
            return f"Error: {e}"


class SkillInstallTool(Tool):
    name = "skill_install"
    description = "Install an external Markdown skill from a local path, SKILL.md directory, or URL into .scicore/skills."
    parameters = {
        "type": "object",
        "properties": {
            "source_path": {"type": "string", "description": "Local path to a .md file or directory containing SKILL.md."},
            "source_url": {"type": "string", "description": "Optional URL to a remote Markdown skill."},
            "slug": {"type": "string", "description": "Optional local slug override."},
            "overwrite": {"type": "boolean", "description": "Overwrite an existing skill. Default false."},
        },
        "required": [],
    }

    def execute(
        self,
        source_path: str | None = None,
        source_url: str | None = None,
        slug: str | None = None,
        overwrite: bool = False,
    ) -> str:
        try:
            if source_url:
                meta = install_skill_from_url(source_url=source_url, slug=slug, overwrite=overwrite)
            elif source_path:
                meta = install_skill_from_path(source_path=source_path, slug=slug, overwrite=overwrite)
            else:
                return "Error: provide source_path or source_url"
            return json.dumps(meta.__dict__, indent=2)
        except Exception as e:
            return f"Error: {e}"


class SkillListTool(Tool):
    name = "skill_list"
    description = "List locally persisted SciCoreAgent skills."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum number of skills to return. Default 20."}
        },
        "required": [],
    }

    def execute(self, limit: int = 20) -> str:
        items = [m.__dict__ for m in list_skills()[:limit]]
        return json.dumps({"skills": items, "count": len(items)}, indent=2)


class SkillReadTool(Tool):
    name = "skill_read"
    description = "Read a locally persisted SciCoreAgent skill by slug."
    parameters = {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Skill slug to read"},
        },
        "required": ["slug"],
    }

    def execute(self, slug: str) -> str:
        try:
            return read_skill(slug)
        except Exception as e:
            return f"Error: {e}"


class SkillSearchTool(Tool):
    name = "skill_search"
    description = "Search locally persisted SciCoreAgent skills by keyword before solving repeat tasks."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword query"},
            "limit": {"type": "integer", "description": "Maximum results. Default 5."},
        },
        "required": ["query"],
    }

    def execute(self, query: str, limit: int = 5) -> str:
        items = [m.__dict__ for m in search_skills(query, limit=limit)]
        return json.dumps({"skills": items, "count": len(items)}, indent=2)
