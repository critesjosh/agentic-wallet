"""Optional inference-time routing skill.

This is an experiment, not part of the frozen contract. When a provider is given
a skill string, it prepends that text to the model prompt as additional
guidance. It is off by default, so production behavior and the trained prompt
are unchanged unless a run explicitly opts in.

The skill only rephrases and disambiguates actions the model is already told
about through ``available_actions`` and ``action_descriptions``. It never adds a
new capability, so it cannot widen the model's reach past deterministic
validation.

Files follow the AI Edge Gallery ``SKILL.md`` shape so they are compatible with
the on-device runtime this project targets: a ``---`` frontmatter block with
``name`` and ``description``, then the instruction body. Only the body is ever
injected into a prompt; the metadata exists for discovery and, in a
progressive-disclosure runtime, gating on relevance before the body is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SKILL_PATH = Path(__file__).resolve().parent / "SKILL.md"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading ``---`` frontmatter block from the body.

    A minimal ``key: value`` reader, so no YAML dependency is added. A file with
    no frontmatter yields empty metadata and the whole text as the body.
    """

    if not text.startswith("---"):
        return {}, text.strip()
    lines = text.splitlines()
    end = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None
    )
    if end is None:
        return {}, text.strip()
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip()
    return metadata, body


def parse_skill(path: str | Path | None = None) -> Skill:
    """Return the parsed skill, or raise if the body is missing or empty."""

    resolved = Path(path) if path is not None else DEFAULT_SKILL_PATH
    metadata, body = _parse_frontmatter(resolved.read_text(encoding="utf-8"))
    if not body:
        raise ValueError(f"skill file has no instructions: {resolved}")
    return Skill(
        name=metadata.get("name", resolved.stem),
        description=metadata.get("description", ""),
        body=body,
    )


def load_skill(path: str | Path | None = None) -> str:
    """Return only the instruction body; frontmatter never enters a prompt."""

    return parse_skill(path).body


def apply_skill(request_text: str, skill: str | None) -> str:
    """Prepend the skill body to a prompt, or return the prompt unchanged."""

    if not skill:
        return request_text
    return f"{skill}\n\n{request_text}"
