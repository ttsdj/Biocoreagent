"""Conversation chunking and evidence-preserving memory distillation."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .features.local_working_memory import LocalWorkingMemoryStore
from .features.postgres_memory import PostgresLongTermMemoryStore

_PREFERENCE = re.compile(r"(?:请|希望|偏好|不要|必须|prefer|please|always)\s*(.+)", re.I)
_DECISION = re.compile(r"(?:决定|采用|使用|应当|需要|decided|use|requires?)\s*(.+)", re.I)
_AUTO_BACKEND = object()


class EvidenceDistiller:
    """Conservative, deterministic first-pass distiller.

    It only emits short preference/decision/fact candidates with the source
    chunk attached.  A model-backed reviewer can be introduced later without
    changing the outbox or Postgres contract.
    """

    version = "evidence_rules_v1"

    def distill(self, payload):
        chunk = dict(payload["chunk"])
        metadata = dict(payload.get("metadata", {}))
        candidates = []
        for line in str(chunk.get("content", "")).splitlines():
            match = re.match(r"^\[(?P<role>[^\]]+)\]\s*(?P<text>.+)$", line.strip())
            if not match:
                continue
            role, text = match.group("role").split(":", 1)[0], match.group("text").strip()
            # Preferences are often intentionally short (for example, “请用
            # 中文”), so only reject fragments rather than short directives.
            if len(text) < 6 or len(text) > 420 or self._looks_sensitive(text):
                continue
            preference = _PREFERENCE.search(text) if role == "user" else None
            decision = _DECISION.search(text) if role in {"assistant", "user"} else None
            if preference:
                memory_type, statement, tags = "user_preference", text, ["user-preference"]
            elif decision:
                memory_type, statement, tags = "decision", text, ["decision"]
            elif role == "assistant" and self._is_stable_fact(text):
                memory_type, statement, tags = "fact", text, ["distilled-fact"]
            else:
                continue
            evidence = {
                "source_uri": chunk["source_uri"],
                "source_sha256": chunk["content_sha256"],
                "session_id": chunk["session_id"],
                "start_message_index": chunk["start_message_index"],
                "end_message_index": chunk["end_message_index"],
            }
            # Identity is the scoped canonical statement, not the current
            # chunk boundary.  A fact may accumulate several source chunks as
            # evidence without becoming duplicate long-term memories.
            canonical = "\0".join(
                [
                    memory_type,
                    statement,
                    str(metadata.get("user_scope", "")),
                    str(metadata.get("project_scope", "")),
                ]
            )
            candidates.append(
                {
                    "memory_id": "mem_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24],
                    "memory_type": memory_type,
                    "statement": statement,
                    "tags": tags,
                    "user_scope": str(metadata.get("user_scope", "")),
                    "project_scope": str(metadata.get("project_scope", "")),
                    "reliability": 0.65,
                    "state": "active",
                    "distiller_version": self.version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "evidence": evidence,
                }
            )
        # Preserve order while ensuring a duplicated sentence is not inserted
        # twice if a chunk contains retried assistant output.
        seen = set()
        return [item for item in candidates if not (item["memory_id"] in seen or seen.add(item["memory_id"]))]

    @staticmethod
    def _looks_sensitive(text):
        lowered = text.lower()
        return bool(re.search(r"(?:api[_-]?key|password|secret|token)\s*[:=]", lowered))

    @staticmethod
    def _is_stable_fact(text):
        return any(marker in text.lower() for marker in (" is ", " are ", " requires ", "使用", "需要", "采用"))


class MemoryPipeline:
    def __init__(self, session_store, *, backend=_AUTO_BACKEND, distiller=None, workspace_root=None):
        self.session_store = session_store
        # Omitted means “discover configured Postgres”; explicit None means
        # “stay offline”, which is useful for isolated workers and tests.
        self.backend = PostgresLongTermMemoryStore.from_environment() if backend is _AUTO_BACKEND else backend
        self.distiller = distiller or EvidenceDistiller()
        local_root = Path(workspace_root) / ".scicore" / "memory" if workspace_root else Path(session_store.root).parent / "memory"
        self.local_store = LocalWorkingMemoryStore(local_root)
        self._drain_lock = threading.RLock()
        self._drain_thread = None
        self._last_async_status = {
            "status": "idle",
            "processed": 0,
            "local_processed": 0,
            "rejected": 0,
            "failed": 0,
            "queued": len(self.session_store.pending_distillations()),
        }

    @property
    def postgres_enabled(self):
        return self.backend is not None

    def capture(self, session, *, metadata=None):
        chunks = self.session_store.create_chunks(session["id"])
        return self.session_store.enqueue_distillation(chunks, metadata=metadata)

    def drain(self, limit=50):
        processed = 0
        local_processed = 0
        rejected = 0
        failed = 0
        for job in self.session_store.pending_distillations(limit):
            try:
                candidates = self.distiller.distill(job["payload"])
                if not candidates:
                    # Without a remote backend, retain even currently
                    # uninteresting chunks: a later model-based distiller may
                    # legitimately extract a stable fact from them.
                    if self.backend is None:
                        continue
                    self.session_store.mark_distillation(job["outbox_id"], status="rejected", error="no_stable_memory")
                    rejected += 1
                    continue
                for candidate in candidates:
                    self.local_store.upsert(candidate)
                local_processed += 1
                if self.backend is None:
                    continue
                for candidate in candidates:
                    self.backend.upsert(candidate)
                    graph_upsert = getattr(self.backend, "upsert_graph_memory", None)
                    if graph_upsert is not None:
                        graph_upsert(candidate)
                self.session_store.mark_distillation(job["outbox_id"], status="processed")
                processed += 1
            except Exception as exc:  # keep retries durable and inspectable
                # Leave failed jobs pending so a temporarily unavailable
                # Postgres/embedding service is retried on the next finished
                # session.  The attempt counter and error remain auditable.
                self.session_store.mark_distillation(job["outbox_id"], status="pending", error=str(exc)[:500])
                failed += 1
        return {
            "processed": processed,
            "local_processed": local_processed,
            "rejected": rejected,
            "failed": failed,
            "queued": len(self.session_store.pending_distillations(limit)),
        }

    def drain_async(self, limit=50):
        """Trigger outbox processing without adding embedding/DB latency to delivery."""
        with self._drain_lock:
            if self._drain_thread is not None and self._drain_thread.is_alive():
                return dict(self._last_async_status)
            self._last_async_status = {
                **self._last_async_status,
                "status": "running",
                "queued": len(self.session_store.pending_distillations(limit)),
            }
            self._drain_thread = threading.Thread(
                target=self._drain_background,
                args=(int(limit),),
                name="biocore-memory-distillation",
                daemon=True,
            )
            self._drain_thread.start()
            return dict(self._last_async_status)

    def async_status(self):
        with self._drain_lock:
            return dict(self._last_async_status)

    def close(self, timeout=5.0):
        thread = self._drain_thread
        if thread is None:
            return self.async_status()
        deadline = time.time() + max(0.0, float(timeout))
        thread.join(timeout=max(0.0, deadline - time.time()))
        return self.async_status()

    def _drain_background(self, limit):
        try:
            result = self.drain(limit)
            status = "completed" if not result["failed"] else "retry_pending"
            with self._drain_lock:
                self._last_async_status = {"status": status, **result}
        except Exception as exc:  # pragma: no cover - defensive thread boundary
            with self._drain_lock:
                self._last_async_status = {
                    **self._last_async_status,
                    "status": "failed",
                    "error": str(exc)[:500],
                }
