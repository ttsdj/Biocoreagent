"""SciCoreAgent: research harness plus bioinformatics domain capabilities."""

from __future__ import annotations

import sys
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

_loaded_scicore_agent = sys.modules.get("scicore.agent")
_loaded_path = Path(getattr(_loaded_scicore_agent, "__file__", "")).resolve() if _loaded_scicore_agent else None
if _loaded_path and _PACKAGE_ROOT not in _loaded_path.parents:
    for _module_name in list(sys.modules):
        if _module_name == "scicore.agent" or _module_name.startswith("scicore.agent."):
            del sys.modules[_module_name]

from .runtime import BioRuntime
from .lifecycle import JobStatus, TeamStatus
from .orchestrator import AsyncMultiAgentOrchestrator, ReplanNode, ReplanPatch

__all__ = [
    "AsyncMultiAgentOrchestrator",
    "BioRuntime",
    "JobStatus",
    "ReplanNode",
    "ReplanPatch",
    "TeamStatus",
]
