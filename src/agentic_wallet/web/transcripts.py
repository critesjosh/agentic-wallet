"""Bounded, process-local transcripts for opt-in development debugging."""

from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class TranscriptTurn:
    sequence: int
    recorded_at: str
    user_message: str
    assistant_reply: str
    workflow_state: str
    data: Any = None


class DebugTranscriptStore:
    """Thread-safe in-memory store with bounded sessions and turns."""

    def __init__(self, *, max_sessions: int = 25, max_turns_per_session: int = 100):
        if max_sessions <= 0 or max_turns_per_session <= 0:
            raise ValueError("transcript limits must be positive")
        self.max_sessions = max_sessions
        self.max_turns_per_session = max_turns_per_session
        self._sessions: OrderedDict[str, deque[TranscriptTurn]] = OrderedDict()
        self._sequence = 0
        self._lock = Lock()

    def record(self, session_id: str, user_message: str, response: dict) -> None:
        with self._lock:
            self._sequence += 1
            turns = self._sessions.setdefault(
                session_id, deque(maxlen=self.max_turns_per_session)
            )
            turns.append(
                TranscriptTurn(
                    sequence=self._sequence,
                    recorded_at=datetime.now(UTC).isoformat(),
                    user_message=user_message,
                    assistant_reply=str(response.get("reply", "")),
                    workflow_state=str(response.get("state", "")),
                    data=deepcopy(response.get("data")),
                )
            )
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def snapshot(self) -> dict:
        with self._lock:
            sessions = [
                {
                    "session_id": session_id,
                    "turns": [asdict(turn) for turn in turns],
                }
                for session_id, turns in reversed(self._sessions.items())
            ]
        return {
            "storage": "process-memory-only",
            "max_sessions": self.max_sessions,
            "max_turns_per_session": self.max_turns_per_session,
            "sessions": sessions,
        }

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


debug_transcripts = DebugTranscriptStore()
