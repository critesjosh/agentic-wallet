"""Optional inference-time routing skill.

This is an experiment, not part of the frozen contract. When a provider is given
a skill string, it prepends that text to the model prompt as additional
guidance. It is off by default, so production behavior and the trained prompt
are unchanged unless a run explicitly opts in.

The skill only rephrases and disambiguates actions the model is already told
about through ``available_actions`` and ``action_descriptions``. It never adds a
new capability, so it cannot widen the model's reach past deterministic
validation.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SKILL_PATH = Path(__file__).resolve().parent / "SKILL.md"

def load_skill(path: str | Path | None = None) -> str:
    """Return the skill text, or raise if the file is missing or empty."""

    resolved = Path(path) if path is not None else DEFAULT_SKILL_PATH
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"skill file is empty: {resolved}")
    return text


def apply_skill(request_text: str, skill: str | None) -> str:
    """Prepend the skill to a prompt, or return the prompt unchanged."""

    if not skill:
        return request_text
    return f"{skill}\n\n{request_text}"
