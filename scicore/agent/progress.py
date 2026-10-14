"""Progress tracking primitives for background agent work."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressState:
    percent: int = 0
    label: str = "queued"
    current_step: str = ""
    total_steps: int | None = None
    completed_steps: int = 0
    updated_at: float = field(default_factory=time.time)

    def update(
        self,
        *,
        percent: int | None = None,
        label: str | None = None,
        current_step: str | None = None,
        total_steps: int | None = None,
        completed_steps: int | None = None,
    ) -> None:
        if percent is not None:
            self.percent = max(0, min(100, int(percent)))
        if label is not None:
            self.label = label
        if current_step is not None:
            self.current_step = current_step
        if total_steps is not None:
            self.total_steps = total_steps
        if completed_steps is not None:
            self.completed_steps = max(0, completed_steps)
        self.updated_at = time.time()

    def bar(self, width: int = 20) -> str:
        filled = int(width * self.percent / 100)
        return "[" + "#" * filled + "-" * (width - filled) + f"] {self.percent}%"

    def to_dict(self) -> dict:
        return {
            "percent": self.percent,
            "label": self.label,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "updated_at": self.updated_at,
            "bar": self.bar(),
        }
