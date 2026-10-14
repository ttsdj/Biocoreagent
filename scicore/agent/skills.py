"""Persistent local skill store for SciCoreAgent.

Skills are small Markdown documents that capture reusable procedures learned
from completed work. They are intentionally plain files so users can review,
edit, version, or delete them without a database.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SKILLS_DIR = Path(".scicore") / "skills"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


@dataclass(frozen=True)
class SkillMetadata:
    slug: str
    title: str
    summary: str
    tags: list[str]
    path: str
    updated_at: str
    version: int = 1


def skills_dir(root: str | Path | None = None) -> Path:
    return Path(root or ".").resolve() / DEFAULT_SKILLS_DIR


def validate_slug(slug: str) -> str:
    normalized = slug.strip().lower()
    if not _SLUG_RE.match(normalized):
        raise ValueError("slug must be 2-64 chars: lowercase letters, numbers, '-' or '_'")
    if ".." in normalized or "/" in normalized or "\\" in normalized:
        raise ValueError("slug must not contain path separators")
    return normalized


def save_skill(
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
    root: str | Path | None = None,
) -> SkillMetadata:
    slug = validate_slug(slug)
    base = skills_dir(root)
    path = (base / f"{slug}.md").resolve()
    if base.resolve() not in path.parents:
        raise ValueError("skill path escaped the skills directory")
    if path.exists() and not overwrite:
        raise FileExistsError(f"skill already exists: {slug}")

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    version = 1
    if path.exists():
        existing = _metadata_from_file(path)
        version = max(1, int(existing.version)) + 1
        _archive_skill_version(path, existing.version)
    clean_tags = _normalize_tags(tags or [])
    content = _render_skill(
        slug=slug,
        title=title.strip(),
        summary=summary.strip(),
        trigger=trigger.strip(),
        procedure=procedure.strip(),
        pitfalls=pitfalls.strip(),
        interview_questions=interview_questions.strip(),
        source_task=source_task.strip(),
        tags=clean_tags,
        updated_at=now,
        version=version,
    )
    base.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return SkillMetadata(slug, title.strip(), summary.strip(), clean_tags, str(path), now, version)


def install_skill_from_path(
    source_path: str | Path,
    slug: str | None = None,
    overwrite: bool = False,
    root: str | Path | None = None,
) -> SkillMetadata:
    """Install an external Markdown skill into the local skill directory."""
    source = Path(source_path).expanduser().resolve()
    if source.is_dir():
        candidate = source / "SKILL.md"
        if not candidate.exists():
            markdown = sorted(source.glob("*.md"))
            if not markdown:
                raise FileNotFoundError("directory does not contain SKILL.md or any .md file")
            candidate = markdown[0]
        source = candidate
    if not source.exists():
        raise FileNotFoundError(f"skill source not found: {source}")
    if source.suffix.lower() != ".md":
        raise ValueError("skill source must be a Markdown file or directory containing SKILL.md")

    text = source.read_text(encoding="utf-8")
    front = _parse_front_matter(text)
    skill_slug = validate_slug(slug or front.get("slug") or _slug_from_title(front.get("title") or source.stem))
    base = skills_dir(root)
    target = (base / f"{skill_slug}.md").resolve()
    if base.resolve() not in target.parents:
        raise ValueError("skill path escaped the skills directory")
    if target.exists() and not overwrite:
        raise FileExistsError(f"skill already exists: {skill_slug}")

    base.mkdir(parents=True, exist_ok=True)
    if front.get("slug"):
        target.write_text(text, encoding="utf-8")
    else:
        title = front.get("title") or _title_from_slug(skill_slug)
        summary = front.get("summary") or first_nonempty_line(text)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        target.write_text(_with_front_matter(text, skill_slug, title, summary, now), encoding="utf-8")
    return _metadata_from_file(target)


def install_skill_from_url(
    source_url: str,
    slug: str | None = None,
    overwrite: bool = False,
    root: str | Path | None = None,
) -> SkillMetadata:
    """Download and install a remote Markdown skill."""
    if not source_url.startswith(("https://", "http://")):
        raise ValueError("source_url must start with https:// or http://")
    with urllib.request.urlopen(source_url, timeout=20) as response:
        raw = response.read(1_000_000)
    text = raw.decode("utf-8")
    front = _parse_front_matter(text)
    skill_slug = validate_slug(slug or front.get("slug") or _slug_from_title(front.get("title") or "remote-skill"))
    base = skills_dir(root)
    target = (base / f"{skill_slug}.md").resolve()
    if base.resolve() not in target.parents:
        raise ValueError("skill path escaped the skills directory")
    if target.exists() and not overwrite:
        raise FileExistsError(f"skill already exists: {skill_slug}")

    base.mkdir(parents=True, exist_ok=True)
    if front.get("slug"):
        target.write_text(text, encoding="utf-8")
    else:
        title = front.get("title") or _title_from_slug(skill_slug)
        summary = front.get("summary") or first_nonempty_line(text)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        target.write_text(_with_front_matter(text, skill_slug, title, summary, now), encoding="utf-8")
    return _metadata_from_file(target)


def list_skills(root: str | Path | None = None) -> list[SkillMetadata]:
    base = skills_dir(root)
    if not base.exists():
        return []
    items = []
    for path in sorted(base.glob("*.md")):
        items.append(_metadata_from_file(path))
    return items


def read_skill(slug: str, root: str | Path | None = None) -> str:
    slug = validate_slug(slug)
    path = (skills_dir(root) / f"{slug}.md").resolve()
    if not path.exists():
        raise FileNotFoundError(f"skill not found: {slug}")
    return path.read_text(encoding="utf-8")


def search_skills(query: str, root: str | Path | None = None, limit: int = 5) -> list[SkillMetadata]:
    terms = [t.lower() for t in re.findall(r"[\w-]+", query) if t.strip()]
    if not terms:
        return list_skills(root)[:limit]
    scored = []
    for meta in list_skills(root):
        text = read_skill(meta.slug, root).lower()
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, meta.updated_at, meta))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [meta for _, _, meta in scored[:limit]]


def skills_prompt_context(root: str | Path | None = None, limit: int = 8, max_chars: int = 4000) -> str:
    """Return a compact prompt section describing installed local skills."""
    metas = list_skills(root)[:limit]
    if not metas:
        return "No local skills installed."
    lines = []
    for meta in metas:
        tag_text = f" tags={','.join(meta.tags)}" if meta.tags else ""
        lines.append(f"- {meta.slug}: {meta.title}. {meta.summary}{tag_text}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... local skill list truncated ..."
    return text


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip().lstrip("#").strip()
        if clean and clean != "---" and ":" not in clean[:30]:
            return clean[:160]
    return "Imported external skill."


def _render_skill(
    slug: str,
    title: str,
    summary: str,
    trigger: str,
    procedure: str,
    pitfalls: str,
    interview_questions: str,
    source_task: str,
    tags: list[str],
    updated_at: str,
    version: int,
) -> str:
    tag_text = ", ".join(tags)
    sections = [
        "---",
        f"slug: {slug}",
        f"title: {title}",
        f"summary: {summary}",
        f"tags: {tag_text}",
        f"version: {version}",
        f"updated_at: {updated_at}",
        "---",
        "",
        f"# {title}",
        "",
        "## When To Use",
        trigger or "Describe the task pattern that should trigger this skill.",
        "",
        "## Procedure",
        procedure or "Document the reusable steps here.",
        "",
        "## Pitfalls",
        pitfalls or "None recorded yet.",
        "",
        "## Interview Notes",
        interview_questions or "No interview notes recorded yet.",
        "",
        "## Source Task",
        source_task or "Not recorded.",
        "",
    ]
    return "\n".join(sections)


def _metadata_from_file(path: Path) -> SkillMetadata:
    text = path.read_text(encoding="utf-8")
    front = _parse_front_matter(text)
    slug = front.get("slug") or path.stem
    title = front.get("title") or slug
    summary = front.get("summary") or ""
    tags = _normalize_tags((front.get("tags") or "").split(","))
    updated_at = front.get("updated_at") or ""
    try:
        version = max(1, int(front.get("version") or 1))
    except ValueError:
        version = 1
    return SkillMetadata(slug, title, summary, tags, str(path.resolve()), updated_at, version)


def _archive_skill_version(path: Path, version: int) -> Path:
    archive_dir = path.parent / ".history" / path.stem
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"v{max(1, int(version))}_{stamp}.md"
    counter = 1
    while archive_path.exists():
        archive_path = archive_dir / f"v{max(1, int(version))}_{stamp}_{counter}.md"
        counter += 1
    archive_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return archive_path


def _parse_front_matter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def _normalize_tags(tags: list[str]) -> list[str]:
    clean = []
    seen = set()
    for tag in tags:
        normalized = re.sub(r"[^a-z0-9_-]+", "-", tag.strip().lower()).strip("-")
        if normalized and normalized not in seen:
            clean.append(normalized)
            seen.add(normalized)
    return clean


def _slug_from_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", title.strip().lower()).strip("-")
    if len(slug) < 2:
        slug = "imported-skill"
    return slug[:64]


def _title_from_slug(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _with_front_matter(text: str, slug: str, title: str, summary: str, updated_at: str) -> str:
    if text.startswith("---\n"):
        return text
    return "\n".join(
        [
            "---",
            f"slug: {slug}",
            f"title: {title}",
            f"summary: {summary}",
            "tags: imported",
            "version: 1",
            f"updated_at: {updated_at}",
            "---",
            "",
            text.rstrip(),
            "",
        ]
    )
