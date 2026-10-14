"""Create human-reviewable cross-session memory benchmark drafts from chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scicore.kernel.session_store import SessionStore


def build_drafts(chunks):
    drafts = []
    for chunk in chunks:
        drafts.append(
            {
                "case_id": "draft_" + str(chunk["chunk_id"]),
                "status": "draft",
                "query": "",
                "relevant_memory_ids": [],
                "forbidden_memory_ids": [],
                "source_uri": str(chunk["source_uri"]),
                "source_sha256": str(chunk["content_sha256"]),
                "review_instructions": (
                    "Write a paraphrased future query, then label the relevant "
                    "Postgres memory IDs and any cross-user IDs that must never return."
                ),
            }
        )
    return drafts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", required=True, help="Path containing sessions.sqlite.")
    parser.add_argument("--output", required=True, help="New JSONL draft path; must not already exist.")
    parser.add_argument("--session-id", default="", help="Optionally export one session.")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; keep reviewed drafts immutable and choose a new path")
    store = SessionStore(args.sessions_root)
    chunks = store.list_chunks(session_id=args.session_id or None, limit=args.limit)
    if not chunks:
        # Migrate a legacy JSON/payload-only session into normalized messages
        # and chunks on demand.  Existing session history is preserved.
        session_ids = [args.session_id] if args.session_id else store.session_ids(limit=args.limit)
        for session_id in session_ids:
            store.save(store.load(session_id))
            store.create_chunks(session_id)
        chunks = store.list_chunks(session_id=args.session_id or None, limit=args.limit)
    if not chunks:
        parser.error("no conversation chunks found; run at least one completed session first")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build_drafts(chunks)),
        encoding="utf-8",
    )
    print(json.dumps({"draft_count": len(chunks), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
