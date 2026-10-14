"""Skill loader for bioinformatics skills.

Loads skill Markdown files distributed with the package (under
scicore/agent/bio/skills/).  These are static reference skills — distinct
from the runtime user skills managed by scicore.agent.skills.

The loader is placed in scicore/agent/bio/ to avoid shadowing the
scicore/agent/skills.py module.
"""

from __future__ import annotations

import re
from pathlib import Path


class SkillLoader:
    """Load and detect bioinformatics skill Markdown files.

    Skills live as .md files in a directory.  The loader reads them on
    demand and can suggest relevant skills for a user query via simple
    keyword matching.
    """

    def __init__(self, skills_path: str | Path | None = None):
        # Default to the skills/ directory next to this file.
        if skills_path is None:
            skills_path = Path(__file__).parent / "skills"
        self._path = Path(skills_path).resolve()

    def load_skill(self, name: str) -> str:
        """Read the full Markdown content of a skill by name (without .md)."""
        path = self._path / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"bio skill not found: {name} (looked in {self._path})")
        return path.read_text(encoding="utf-8")

    def list_skills(self) -> list[str]:
        """Return available skill names (sorted, without .md extension)."""
        if not self._path.exists():
            return []
        return sorted(p.stem for p in self._path.glob("*.md"))

    def detect_relevant_skills(
        self, query: str, limit: int = 3
    ) -> list[str]:
        """Return skill names relevant to *query*, best match first.

        Phase 1: simple keyword frequency scoring (same approach as
        skills.py:search_skills and evidence_store.py:EvidenceStore.search).
        Each word in *query* is counted against the lowercased skill content;
        skills with any match are ranked by total term frequency.

        Keyword triggers (Chinese + English):
        - protocol / methods / paper / reproduce / 方案 / 方法 / 实验方案
          → protocol_extraction, replication_planning
        - workflow / snakemake / nextflow / pipeline / 流程 / 脚本
          → workflow_generation
        """
        terms = [t.lower() for t in re.findall(r"[\w-]+", query) if t.strip()]
        if not terms:
            return self.list_skills()[:limit]

        scored: list[tuple[int, str]] = []
        for name in self.list_skills():
            try:
                text = self.load_skill(name).lower()
            except FileNotFoundError:
                continue
            score = sum(text.count(term) for term in terms)
            if score:
                scored.append((score, name))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in scored[:limit]]
