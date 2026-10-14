"""Read-only review helper for memory-recall draft cases and their evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scicore.kernel.session_store import SessionStore


_SOURCE_URI = re.compile(r"^session://(?P<session>[^/]+)/messages/(?P<start>\d+)-(?P<end>\d+)#sha256=[0-9a-f]+$")


def parse_source_uri(value):
    match = _SOURCE_URI.match(str(value))
    if not match:
        raise ValueError(f"unsupported source URI: {value}")
    return match.group("session"), int(match.group("start")), int(match.group("end"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", help="Draft JSONL created by prepare_memory_recall_cases.py")
    parser.add_argument("--sessions-root", required=True, help="Path containing sessions.sqlite")
    parser.add_argument("--case-id", default="", help="Show only one draft case")
    args = parser.parse_args(argv)
    store = SessionStore(args.sessions_root)
    matched = 0
    for line in Path(args.drafts).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        draft = json.loads(line)
        if args.case_id and str(draft.get("case_id", "")) != args.case_id:
            continue
        session_id, start, end = parse_source_uri(draft["source_uri"])
        messages = store.source_excerpt(session_id, start, end)
        print(json.dumps({
            "case_id": draft.get("case_id"),
            "status": draft.get("status"),
            "source_uri": draft["source_uri"],
            "messages": messages,
            "next": "Author an independent query and label relevant/forbidden memory IDs in the draft, then mark it ready.",
        }, ensure_ascii=False, indent=2))
        matched += 1
    if not matched:
        parser.error("no matching draft cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
