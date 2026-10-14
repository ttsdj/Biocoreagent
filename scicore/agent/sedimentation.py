"""Background review that turns repeated successful workflows into versioned draft skills."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .skills import list_skills, read_skill, save_skill
from .wiki import save_entry


REVIEW_THRESHOLD = 3
FAILURE_MARKERS = (
    "analysis failed",
    "analysis not completed",
    "recovery mode",
    "未完成",
    "执行失败",
    "已停止",
    "traceback",
)
_LOCK = threading.RLock()


DREAM_STAGES = (
    "detect",
    "review",
    "extract",
    "apply",
    "monitor",
)


class SedimentationReviewer:
    def __init__(self, root: str | Path, threshold: int = REVIEW_THRESHOLD):
        self.root = Path(root).resolve()
        self.threshold = max(2, int(threshold))
        self.review_dir = self.root / ".scicore" / "sedimentation"
        self.observations_path = self.review_dir / "observations.jsonl"

    def review(
        self,
        user_message: str,
        final_answer: str,
        tool_events: list[dict[str, Any]],
        *,
        source_session: str = "",
    ) -> list[dict[str, Any]]:
        if not self._eligible(final_answer, tool_events):
            return []
        topic = _topic_key(user_message)
        tool_sequence = [
            str(item.get("name", "")).strip()
            for item in tool_events
            if str(item.get("name", "")).strip()
        ]
        observation = {
            "observation_id": _observation_id(user_message, tool_sequence, final_answer),
            "topic": topic,
            "request": str(user_message)[:1000],
            "tool_sequence": tool_sequence,
            "answer_summary": str(final_answer)[:1200],
            "source_session": str(source_session),
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        with _LOCK:
            observations = self._load_observations()
            if any(
                item.get("observation_id") == observation["observation_id"]
                for item in observations
            ):
                return []
            self.review_dir.mkdir(parents=True, exist_ok=True)
            with self.observations_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(observation, ensure_ascii=False) + "\n")
            observations.append(observation)

            topic_observations = [
                item for item in observations if item.get("topic") == topic
            ]
            generation = len(topic_observations) // self.threshold
            if generation < 1:
                return []

            slug = f"learned-{topic}"[:64]
            existing = next(
                (item for item in list_skills(self.root) if item.slug == slug),
                None,
            )
            if existing is not None and existing.version >= generation:
                return []

            action = "updated" if existing is not None else "created"
            skill = self._write_skill(
                slug,
                topic,
                topic_observations,
                overwrite=existing is not None,
            )
            wiki = save_entry(
                title=f"Sedimentation review: {topic} v{skill.version}",
                content=(
                    f"Observed {len(topic_observations)} successful runs for `{topic}`.\n\n"
                    f"Draft skill: `{skill.path}`\n\n"
                    "This entry records the evidence boundary; the skill remains tagged "
                    "`draft` until a human reviews it."
                ),
                tags=["sedimentation", "skill-review", topic],
                source_session=str(source_session),
                root=self.root,
            )
            return [
                {
                    "kind": "background_skill_review",
                    "action": action,
                    "slug": skill.slug,
                    "version": skill.version,
                    "observations": len(topic_observations),
                    "path": skill.path,
                    "wiki_id": wiki["id"],
                }
            ]

    def _eligible(
        self,
        final_answer: str,
        tool_events: list[dict[str, Any]],
    ) -> bool:
        if not tool_events:
            return False
        lowered = str(final_answer).lower()
        if not str(final_answer).strip():
            return False
        if any(marker in lowered for marker in FAILURE_MARKERS):
            return False
        statuses = [
            str(item.get("metadata", {}).get("tool_status", "")).strip()
            for item in tool_events
        ]
        return not any(status in {"error", "rejected"} for status in statuses)

    def _load_observations(self) -> list[dict[str, Any]]:
        if not self.observations_path.exists():
            return []
        rows = []
        for line in self.observations_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _write_skill(
        self,
        slug: str,
        topic: str,
        observations: list[dict[str, Any]],
        *,
        overwrite: bool,
    ):
        sequences = Counter(
            " -> ".join(item.get("tool_sequence", []))
            for item in observations
            if item.get("tool_sequence")
        )
        observed_sequences = [
            f"- `{sequence}` ({count} runs)"
            for sequence, count in sequences.most_common(5)
        ]
        procedure = "\n".join(
            [
                "1. Confirm the task inputs and workspace boundary.",
                "2. Select the registered deterministic capability before free-form execution.",
                "3. Follow one of the observed successful tool sequences:",
                *observed_sequences,
                "4. Verify output artifacts and report unresolved blockers explicitly.",
            ]
        )
        if overwrite:
            previous = read_skill(slug, self.root)
            previous_note = (
                "\n\nPrevious version was archived automatically before this update. "
                f"Previous content hash: sha256:{hashlib.sha256(previous.encode('utf-8')).hexdigest()}."
            )
            procedure += previous_note
        return save_skill(
            slug=slug,
            title=f"Learned {topic.replace('-', ' ').title()} Workflow",
            summary=(
                f"Draft workflow distilled from {len(observations)} successful local runs; "
                "human review required before trusted reuse."
            ),
            trigger=f"Use for repeated `{topic}` tasks after checking inputs and registered capabilities.",
            procedure=procedure,
            pitfalls=(
                "Automatically distilled evidence can preserve a locally successful but "
                "non-general workflow. Review data assumptions, versions, and verifier output."
            ),
            interview_questions=(
                "Why is the auto-generated skill marked draft? How are versions and source "
                "observations preserved?"
            ),
            source_task="\n".join(
                f"- {item.get('request', '')[:300]}"
                for item in observations[-self.threshold :]
            ),
            tags=["auto", "draft", "reviewed", topic],
            overwrite=overwrite,
            root=self.root,
        )


class DreamSedimentationManager:
    """Durable asynchronous skill-sedimentation queue.

    DREAM means Detect → Review → Extract → Apply → Monitor.  ``submit`` only
    persists the evidence payload and enqueues work; a daemon worker performs
    the conservative reviewer pass.  Job files make pending/running/completed
    work inspectable and allow unfinished jobs to resume after a clean restart.
    """

    def __init__(self, root: str | Path, *, threshold: int = REVIEW_THRESHOLD):
        self.root = Path(root).resolve()
        self.reviewer = SedimentationReviewer(self.root, threshold=threshold)
        self.dream_dir = self.reviewer.review_dir / "dream"
        self.jobs_dir = self.dream_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"scicore-dream-{self.root.name}",
            daemon=True,
        )
        self._recover_unfinished()
        self._thread.start()

    def submit(
        self,
        user_message: str,
        final_answer: str,
        tool_events: list[dict[str, Any]],
        *,
        source_session: str = "",
    ) -> dict[str, Any]:
        job_id = "dream_" + uuid.uuid4().hex[:16]
        job = {
            "job_id": job_id,
            "status": "queued",
            "stage": DREAM_STAGES[0],
            "stages": {
                stage: "pending"
                for stage in DREAM_STAGES
            },
            "payload": {
                "user_message": str(user_message),
                "final_answer": str(final_answer),
                "tool_events": list(tool_events),
                "source_session": str(source_session),
            },
            "result": [],
            "error": "",
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        self._write_job(job)
        self._queue.put(job_id)
        return self._public(job)

    def status(self, job_id: str) -> dict[str, Any] | None:
        path = self.jobs_dir / f"{str(job_id)}.json"
        if not path.exists():
            return None
        with self._lock:
            return self._public(json.loads(path.read_text(encoding="utf-8")))

    def wait(self, job_id: str, timeout: float = 5.0) -> dict[str, Any] | None:
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() < deadline:
            current = self.status(job_id)
            if current is None or current["status"] in {"completed", "failed", "rejected"}:
                return current
            time.sleep(0.02)
        return self.status(job_id)

    def close(self, timeout: float = 5.0) -> None:
        deadline = time.time() + max(0.0, float(timeout))
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.02)
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=max(0.0, deadline - time.time()))

    def _recover_unfinished(self) -> None:
        for path in sorted(self.jobs_dir.glob("dream_*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in {"queued", "running"}:
                job["status"] = "queued"
                job["error"] = ""
                self._write_job(job)
                self._queue.put(str(job["job_id"]))

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._process(job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return
        with self._lock:
            job = json.loads(path.read_text(encoding="utf-8"))
        try:
            self._advance(job, "detect")
            payload = dict(job["payload"])
            eligible = self.reviewer._eligible(
                payload["final_answer"],
                payload["tool_events"],
            )
            if not eligible:
                for stage in DREAM_STAGES[1:]:
                    job["stages"][stage] = "skipped"
                job["status"] = "rejected"
                job["error"] = "ineligible_or_unverified_run"
                self._write_job(job)
                return

            self._advance(job, "review")
            result = self.reviewer.review(
                payload["user_message"],
                payload["final_answer"],
                payload["tool_events"],
                source_session=payload["source_session"],
            )
            self._advance(job, "extract")
            self._advance(job, "apply")
            job["result"] = result
            self._advance(job, "monitor")
            job["status"] = "completed"
            self._write_job(job)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)[:1000]
            self._write_job(job)

    def _advance(self, job: dict[str, Any], stage: str) -> None:
        job["status"] = "running"
        job["stage"] = stage
        job["stages"][stage] = "completed"
        self._write_job(job)

    def _write_job(self, job: dict[str, Any]) -> None:
        job["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        path = self.jobs_dir / f"{job['job_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(
                json.dumps(job, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if key != "payload"
        }


def _observation_id(
    user_message: str,
    tool_sequence: list[str],
    final_answer: str,
) -> str:
    payload = json.dumps(
        {
            "request": str(user_message),
            "tools": tool_sequence,
            "answer": str(final_answer),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _topic_key(text: str) -> str:
    lowered = str(text).lower()
    routes = (
        ("bulk-rnaseq", ("rna-seq", "rnaseq", "deseq", "count matrix", "转录组", "差异表达")),
        ("literature-review", ("pubmed", "literature", "paper", "文献", "调研")),
        ("proteomics", ("proteomics", "protein intensity", "蛋白组")),
        ("single-cell", ("single cell", "scrna", "h5ad", "单细胞")),
        ("variant-analysis", ("variant", "vcf", "mutation", "变异")),
    )
    for topic, keywords in routes:
        if any(keyword in lowered for keyword in keywords):
            return topic
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", lowered)
        if len(token) >= 3
    ]
    return "-".join(tokens[:3])[:48] or "generic-workflow"
