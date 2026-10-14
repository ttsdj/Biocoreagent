"""Local, append-only evidence and source provenance for SciCoreAgent runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from scicore.kernel.workspace import IGNORED_PATH_NAMES, now


TRACE_SCHEMA_VERSION = 3
MAX_CODE_BYTES = 10 * 1024 * 1024
CODE_SUFFIXES = {
    ".py", ".pyi", ".r", ".rmd", ".qmd", ".ipynb", ".sh", ".bash",
    ".zsh", ".ps1", ".psm1", ".toml", ".yaml", ".yml", ".json",
    ".ini", ".cfg", ".sql", ".jl", ".m", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
}
CODE_NAMES = {
    "Dockerfile", "Makefile", "Snakefile", "environment.yml",
    "requirements.txt", "pyproject.toml", "renv.lock",
}
SECRET_FILE_RE = re.compile(
    r"(?i)(^|[._-])(env|secret|token|password|credential|api[_-]?key)([._-]|$)"
)
_PACKAGE_CACHE: dict[str, str] | None = None
_R_ENV_CACHE: dict[str, Any] | None = None
CAPTURED_PACKAGES = (
    "SciCoreAgent",
    "openai",
    "pydantic",
    "biopython",
    "numpy",
    "pandas",
    "scipy",
    "openpyxl",
    "python-docx",
    "omicverse",
)
CAPTURED_R_PACKAGES = (
    "DESeq2", "edgeR", "limma", "Seurat", "BiocManager", "rlang",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def is_code_path(path: Path) -> bool:
    return path.name in CODE_NAMES or path.suffix.lower() in CODE_SUFFIXES


def safe_relative(path: Path, root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if any(part in IGNORED_PATH_NAMES or part == ".scicore" for part in relative.parts):
        return None
    return relative.as_posix()


class ContentAddressedStore:
    """Deduplicated store for redacted source snapshots."""

    def __init__(self, root: str | Path, redactor: Callable[[str], str] | None = None):
        self.root = Path(root)
        self.redactor = redactor or (lambda value: value)

    def capture(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file() or not is_code_path(path):
            return None
        if SECRET_FILE_RE.search(path.name) or path.stat().st_size > MAX_CODE_BYTES:
            return None
        try:
            original = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        redacted = self.redactor(original)
        data = redacted.encode("utf-8")
        digest = sha256_bytes(data)
        object_path = self.root / "sha256" / digest[:2] / digest
        if not object_path.exists():
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(data)
        return {
            "sha256": digest,
            "object_id": f"sha256:{digest}",
            "size_bytes": len(data),
            "redaction_status": "redacted" if redacted != original else "clean",
        }

    def object_path(self, object_id: str) -> Path:
        algorithm, _, digest = str(object_id).partition(":")
        if algorithm != "sha256" or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid object id: {object_id}")
        return self.root / "sha256" / digest[:2] / digest


class EvidenceManager:
    """Capture source versions, git state, deliverables, and run indexes."""

    def __init__(
        self,
        workspace_root: str | Path,
        state_root: str | Path,
        *,
        redactor: Callable[[str], str] | None = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.state_root = Path(state_root).resolve()
        self.store = ContentAddressedStore(self.state_root / "objects", redactor)
        self.redactor = redactor or (lambda value: value)
        self.run_id = ""
        self.run_dir: Path | None = None
        self.last_event_id = ""
        self._captured_keys: set[tuple[str, str, str]] = set()

    def begin_run(self, run_id: str, run_dir: str | Path, event_id: str) -> None:
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.last_event_id = event_id
        self._captured_keys = set()
        write_json_atomic(self.run_dir / "git_before.json", self.capture_git_state("before"))
        write_json_atomic(self.run_dir / "environment.json", self.capture_environment())
        self.capture_code_tree(version_role="run_start", captured_at_event=event_id)

    def note_event(self, event: dict[str, Any]) -> None:
        self.last_event_id = str(event.get("event_id", self.last_event_id))

    def capture_code_tree(self, *, version_role: str, captured_at_event: str = "") -> list[dict[str, Any]]:
        if self.run_dir is None:
            return []
        captured = []
        for path in self.workspace_root.rglob("*"):
            relative = safe_relative(path, self.workspace_root)
            if relative is None or not is_code_path(path):
                continue
            entry = self.capture_path(
                path,
                version_role=version_role,
                captured_at_event=captured_at_event or self.last_event_id,
            )
            if entry:
                captured.append(entry)
        return captured

    def capture_paths(
        self,
        paths: Iterable[str | Path],
        *,
        version_role: str,
        captured_at_event: str = "",
    ) -> list[dict[str, Any]]:
        captured = []
        for value in paths:
            path = Path(value)
            absolute = path if path.is_absolute() else self.workspace_root / path
            entry = self.capture_path(
                absolute,
                version_role=version_role,
                captured_at_event=captured_at_event or self.last_event_id,
            )
            if entry:
                captured.append(entry)
        return captured

    def capture_path(
        self,
        path: Path,
        *,
        version_role: str,
        captured_at_event: str,
    ) -> dict[str, Any] | None:
        relative = safe_relative(path, self.workspace_root)
        if relative is None:
            return None
        object_entry = self.store.capture(path)
        if object_entry is None:
            return None
        key = (relative, object_entry["sha256"], version_role)
        if key in self._captured_keys:
            return None
        self._captured_keys.add(key)
        entry = {
            "path": relative,
            **object_entry,
            "captured_at": now(),
            "captured_at_event": captured_at_event,
            "version_role": version_role,
        }
        append_jsonl(self.run_dir / "source_manifest.jsonl", entry)
        return entry

    def capture_git_state(self, phase: str) -> dict[str, Any]:
        def git(*arguments: str) -> str:
            try:
                process = subprocess.run(
                    ["git", *arguments],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
                return self.redactor(process.stdout.strip())
            except Exception as exc:
                return f"unavailable: {exc}"

        discovered_root = git("rev-parse", "--show-toplevel")
        try:
            is_repository_root = Path(discovered_root).resolve() == self.workspace_root
        except (OSError, ValueError):
            is_repository_root = False
        if not is_repository_root:
            payload = {
                "captured_at": now(),
                "phase": phase,
                "repo_root": str(self.workspace_root),
                "is_git_repository": False,
                "discovered_parent_repository": discovered_root,
            }
            persisted = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            payload["snapshot_sha256"] = sha256_bytes(persisted)
            return payload

        payload = {
            "captured_at": now(),
            "phase": phase,
            "repo_root": str(self.workspace_root),
            "is_git_repository": True,
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "remote": git("remote", "get-url", "origin"),
            "status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all"),
            "unstaged_patch": git("diff", "--no-ext-diff", "--binary"),
            "staged_patch": git("diff", "--cached", "--no-ext-diff", "--binary"),
        }
        persisted = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        payload["snapshot_sha256"] = sha256_bytes(persisted)
        return payload

    def capture_environment(self) -> dict[str, Any]:
        global _PACKAGE_CACHE, _R_ENV_CACHE
        if _PACKAGE_CACHE is None:
            _PACKAGE_CACHE = {}
            for name in CAPTURED_PACKAGES:
                try:
                    _PACKAGE_CACHE[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    continue
        packages = _PACKAGE_CACHE
        if _R_ENV_CACHE is None:
            _R_ENV_CACHE = _capture_r_environment()
        container_markers = {
            name: os.environ.get(name, "")
            for name in ("CONTAINER", "DOCKER_IMAGE", "KUBERNETES_SERVICE_HOST")
            if os.environ.get(name)
        }
        return {
            "captured_at": now(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
            "r": dict(_R_ENV_CACHE),
            "container": {
                "detected": bool(container_markers or Path("/.dockerenv").exists()),
                "markers": self.redactor(json.dumps(container_markers, ensure_ascii=False)),
            },
            "environment_variable_names": sorted(os.environ),
        }

    def record_tool_completion(
        self,
        *,
        name: str,
        args: dict[str, Any],
        result: Any,
        metadata: dict[str, Any],
        event_id: str,
    ) -> list[dict[str, str]]:
        affected = list(metadata.get("affected_paths", []) or [])
        self.capture_paths(affected, version_role="tool_output", captured_at_event=event_id)
        bundles = []
        for relative in affected:
            path = self.workspace_root / relative
            if path.is_file():
                bundle = self.create_deliverable(
                    path,
                    tool_name=name,
                    tool_args=args,
                    tool_result=result,
                    tool_metadata=metadata,
                    event_id=event_id,
                )
                if bundle is not None:
                    bundles.append(
                        {
                            "artifact_path": relative,
                            "bundle_path": str(bundle),
                        }
                    )
        return bundles

    def create_deliverable(
        self,
        artifact_path: Path,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        tool_metadata: dict[str, Any],
        event_id: str,
    ) -> Path | None:
        if self.run_dir is None:
            return None
        relative = safe_relative(artifact_path, self.workspace_root)
        if relative is None or not artifact_path.is_file():
            return None
        artifact_hash = sha256_file(artifact_path)
        deliverable_id = "deliverable_" + artifact_hash[:12]
        bundle = self.run_dir / "deliverables" / deliverable_id
        bundle.mkdir(parents=True, exist_ok=True)
        source_entries = self.source_entries(up_to_event=event_id)
        manifest = {
            "schema_version": 1,
            "deliverable_id": deliverable_id,
            "artifact": {
                "path": relative,
                "sha256": artifact_hash,
                "size_bytes": artifact_path.stat().st_size,
                "media_type": _media_type(artifact_path),
            },
            "created_at": now(),
            "created_at_event": event_id,
            "tool": tool_name,
            "tool_args": _redact_value(tool_args, self.redactor),
            "tool_result_preview": self.redactor(str(tool_result))[:1000],
            "execution": {
                "command": self.redactor(str(tool_args.get("command", ""))),
                "working_directory": str(self.workspace_root),
                "status": str(tool_metadata.get("tool_status", "")),
                "error_code": str(tool_metadata.get("tool_error_code", "")),
                "execution_id": str(tool_metadata.get("execution_id", "")),
                "duration_ms": int(tool_metadata.get("duration_ms", 0) or 0),
            },
            "inputs": self._input_descriptors(tool_args, exclude=relative),
            "verifier": {"status": "not_run"},
        }
        write_json_atomic(bundle / "manifest.json", manifest)
        environment_path = self.run_dir / "environment.json"
        write_json_atomic(
            bundle / "environment.json",
            json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {},
        )
        write_json_atomic(
            bundle / "lineage.json",
            {
                "trace": str((self.run_dir / "trace.jsonl").resolve()),
                "trace_event": event_id,
                "git_before": str((self.run_dir / "git_before.json").resolve()),
                "source_manifest": str((self.run_dir / "source_manifest.jsonl").resolve()),
            },
        )
        write_json_atomic(bundle / "code_manifest.json", {"sources": source_entries})
        (bundle / "README.md").write_text(
            f"# {deliverable_id}\n\n"
            f"- Artifact: `{relative}`\n"
            f"- SHA-256: `{artifact_hash}`\n"
            f"- Tool: `{tool_name}`\n"
            f"- Trace event: `{event_id}`\n",
            encoding="utf-8",
        )
        return bundle

    def _input_descriptors(self, tool_args: dict[str, Any], *, exclude: str) -> list[dict[str, Any]]:
        descriptors = []
        for key, value in tool_args.items():
            if not isinstance(value, str) or not re.search(
                r"(?i)(path|file|input|source|matrix|metadata|counts|table)",
                str(key),
            ):
                continue
            candidate = Path(value)
            absolute = candidate if candidate.is_absolute() else self.workspace_root / candidate
            relative = safe_relative(absolute, self.workspace_root)
            if relative is None or relative == exclude or not absolute.is_file():
                continue
            descriptors.append(
                {
                    "parameter": str(key),
                    "path": relative,
                    "sha256": sha256_file(absolute),
                    "size_bytes": absolute.stat().st_size,
                    "stored_as_fixture": False,
                }
            )
        return descriptors

    def source_entries(self, *, up_to_event: str = "") -> list[dict[str, Any]]:
        if self.run_dir is None:
            return []
        path = self.run_dir / "source_manifest.jsonl"
        if not path.exists():
            return []
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not up_to_event:
            return rows
        # Event IDs are not sortable; capture time and event linkage remain visible.
        return rows

    def finish_run(self, event_id: str) -> None:
        if self.run_dir is None:
            return
        self.capture_code_tree(version_role="run_end", captured_at_event=event_id)
        write_json_atomic(self.run_dir / "git_after.json", self.capture_git_state("after"))

    def build_evidence_index(self, runtime_identity: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.run_dir is None:
            raise RuntimeError("evidence run has not started")
        files = {}
        for path in sorted(item for item in self.run_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(self.run_dir).as_posix()
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        events = []
        trace = self.run_dir / "trace.jsonl"
        if trace.exists():
            events = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        terminal = [row for row in events if row.get("event_type") == "run_finished" or row.get("event") == "run_finished"]
        identity = dict(runtime_identity or {})
        git_before = self.run_dir / "git_before.json"
        if git_before.is_file():
            git_payload = json.loads(git_before.read_text(encoding="utf-8"))
            identity.update(
                {
                    "git_head": git_payload.get("head", ""),
                    "git_branch": git_payload.get("branch", ""),
                    "git_snapshot_sha256": git_payload.get("snapshot_sha256", ""),
                }
            )
        index = {
            "schema_version": 1,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "workspace_root": str(self.workspace_root),
            "created_at": now(),
            "event_count": len(events),
            "terminal_event_count": len(terminal),
            "trace_complete": len(terminal) == 1,
            "runtime_identity": identity,
            "files": files,
        }
        write_json_atomic(self.run_dir / "evidence_index.json", index)
        return index


def _capture_r_environment() -> dict[str, Any]:
    executable = shutil.which("Rscript")
    if not executable:
        return {"available": False, "executable": "", "version": "", "packages": {}}
    try:
        version_process = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        package_expression = (
            "wanted<-c(" +
            ",".join(f"'{name}'" for name in CAPTURED_R_PACKAGES) +
            "); ip<-installed.packages(); found<-intersect(wanted,rownames(ip)); "
            "if(length(found)) cat(paste(found,ip[found,'Version'],sep='=',collapse='\\n'))"
        )
        package_process = subprocess.run(
            [executable, "-e", package_expression],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        packages = {}
        for line in package_process.stdout.splitlines():
            name, separator, version = line.strip().partition("=")
            if separator and name:
                packages[name] = version
        return {
            "available": True,
            "executable": executable,
            "version": (
                version_process.stdout.strip()
                or version_process.stderr.strip()
            ),
            "packages": packages,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "executable": executable,
            "version": "",
            "packages": {},
            "capture_error": str(exc)[:500],
        }


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "application/octet-stream")


def _redact_value(value: Any, redactor: Callable[[str], str]) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        return json.loads(redactor(encoded))
    except Exception:
        return redactor(str(value))
