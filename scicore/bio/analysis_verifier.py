"""Final-answer consistency guard for SciCoreAgent."""

from __future__ import annotations

import re


FAILURE_MARKERS = (
    "failed",
    "failure",
    "error",
    "traceback",
    "execution halted",
    "exit code",
    "崩溃",
    "失败",
    "报错",
    "尚未生成",
    "未生成",
)


SUCCESS_PATTERNS = (
    r"分析完成",
    r"已完成",
    r"completed",
    r"successfully completed",
)


def verify_final_answer(text: str) -> str:
    answer = str(text or "")
    lowered = answer.lower()
    claims_success = any(re.search(pattern, answer, flags=re.I) for pattern in SUCCESS_PATTERNS)
    has_failure = any(marker in lowered for marker in FAILURE_MARKERS)
    if not (claims_success and has_failure):
        return answer

    rewritten = answer
    rewritten = re.sub(r"分析完成", "分析未完成", rewritten)
    rewritten = re.sub(r"已完成", "未完成", rewritten)
    rewritten = re.sub(r"successfully completed", "not completed", rewritten, flags=re.I)
    rewritten = re.sub(r"\bcompleted\b", "not completed", rewritten, flags=re.I)
    prefix = (
        "最终答案校验：检测到输出同时声称“完成”并包含失败证据，"
        "已按实际执行证据改写为未完成状态。\n\n"
    )
    return prefix + rewritten
