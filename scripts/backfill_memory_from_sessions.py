"""Idempotently enqueue and distill historical local sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scicore.kernel.memory_pipeline import MemoryPipeline
from scicore.kernel.session_store import SessionStore


def backfill_sessions(store, pipeline, session_ids, *, user_scope="", project_scope=""):
    totals = {"sessions": 0, "chunks_enqueued": 0, "processed": 0, "local_processed": 0, "rejected": 0, "failed": 0, "queued": 0}
    for session_id in session_ids:
        session = store.load(session_id)
        totals["sessions"] += 1
        totals["chunks_enqueued"] += pipeline.capture(
            session,
            metadata={"user_scope": user_scope, "project_scope": project_scope},
        )
        result = pipeline.drain()
        for key in ("processed", "local_processed", "rejected", "failed", "queued"):
            totals[key] += int(result.get(key, 0))
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", required=True)
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--user-scope", default="")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    store = SessionStore(args.sessions_root)
    session_ids = [args.session_id] if args.session_id else store.session_ids(limit=args.limit)
    if not session_ids:
        parser.error("no local sessions found")
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else Path(args.sessions_root).resolve().parent.parent
    pipeline = MemoryPipeline(store, workspace_root=workspace_root)
    result = backfill_sessions(
        store,
        pipeline,
        session_ids,
        user_scope=args.user_scope,
        project_scope=str(workspace_root),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
