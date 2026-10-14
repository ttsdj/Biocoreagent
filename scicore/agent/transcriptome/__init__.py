"""Bulk transcriptome capability layer.

This package is the RNA-seq counterpart of an OmicOS-style capability layer:
capabilities declare their required state, produced state, parameters, side
effects, and execution tool, while the planner/verifier reason over those
contracts before any code is executed.
"""

from .planner import plan_transcriptome_workflow
from .registry import CAPABILITIES, get_capability, list_capabilities
from .state import TranscriptomeState, inspect_transcriptome_state
from .verifier import verify_plan

__all__ = [
    "CAPABILITIES",
    "TranscriptomeState",
    "get_capability",
    "inspect_transcriptome_state",
    "list_capabilities",
    "plan_transcriptome_workflow",
    "verify_plan",
]
