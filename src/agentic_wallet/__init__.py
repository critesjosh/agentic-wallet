"""Private on-device crypto agent: deterministic harness and typed schemas.

The model proposes actions; deterministic code enforces and executes. Live
blockchain state comes from typed tools, never from model weights, and the
model never receives wallet secrets. See plan.md (Consensus revisions) for the
architecture this package realizes.
"""

from .state_machine import StateMachine, TransitionError, WorkflowState

__all__ = ["WorkflowState", "StateMachine", "TransitionError"]
__version__ = "0.0.1"
