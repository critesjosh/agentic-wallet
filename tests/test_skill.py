"""The optional inference-time routing skill.

The skill must be strictly additive: off by default, prepended without altering
the frozen contract text, and never a new capability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_wallet.skill import (
    DEFAULT_SKILL_PATH,
    apply_skill,
    load_skill,
    parse_skill,
)

CONTRACT = '{"phase":"route_dialogue","user_request":"what is my address?"}'


def test_bundled_skill_loads_and_stays_terse():
    text = load_skill()

    assert text
    # A mobile context window is tight; keep the skill small. This is a
    # regression guard, not a hard runtime limit.
    assert len(text) < 900, "SKILL.md has grown; keep it terse for on-device use"


def test_frontmatter_is_parsed_and_never_enters_the_body():
    """Match the AI Edge Gallery SKILL.md shape: metadata plus instructions."""

    skill = parse_skill()

    assert skill.name and skill.description
    # The body is the instructions only; the YAML metadata must not leak into it.
    assert "---" not in skill.body
    assert "description:" not in skill.body
    assert skill.body.startswith("Routing")
    # load_skill returns exactly the body.
    assert load_skill() == skill.body


def test_all_bundled_skills_have_metadata():
    from pathlib import Path

    skills_dir = DEFAULT_SKILL_PATH.parent / "skills"
    for path in [DEFAULT_SKILL_PATH, *sorted(skills_dir.glob("*.md"))]:
        skill = parse_skill(path)
        assert skill.name, f"{path} missing name"
        assert skill.description, f"{path} missing description"
        assert "name:" not in skill.body


def test_apply_skill_is_a_noop_when_absent():
    assert apply_skill(CONTRACT, None) == CONTRACT
    assert apply_skill(CONTRACT, "") == CONTRACT


def test_apply_skill_prepends_without_mutating_the_contract():
    skill = load_skill()
    combined = apply_skill(CONTRACT, skill)

    assert combined.endswith(CONTRACT)
    assert skill in combined
    # The contract text is untouched, byte for byte.
    assert combined[-len(CONTRACT):] == CONTRACT


def test_empty_skill_file_is_rejected(tmp_path: Path):
    empty = tmp_path / "SKILL.md"
    empty.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_skill(empty)


def test_default_path_points_at_the_bundled_file():
    assert DEFAULT_SKILL_PATH.name == "SKILL.md"
    assert DEFAULT_SKILL_PATH.is_file()


def test_ollama_provider_injects_skill_as_a_system_message():
    from agentic_wallet.providers import OllamaProvider

    captured: dict = {}

    def transport(url, payload, timeout):
        captured["payload"] = payload
        return {
            "done": True,
            "done_reason": "stop",
            "message": {"content": '{"proposed_action":"get_account"}'},
        }

    provider = OllamaProvider(
        model="gemma4:e2b", transport=transport, skill="ROUTING RULES"
    )
    provider.propose_dialogue_route(
        {"user_request": "what is my address?"},
        ["get_account", "get_portfolio"],
        ["get_portfolio"],
    )

    messages = captured["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "ROUTING RULES"}
    # The skill is additive: the original contract messages remain present.
    assert len(messages) >= 2


def test_routing_skill_does_not_reach_the_argument_phase():
    """Routing guidance in the argument phase broke JSON construction; scope it out."""

    from agentic_wallet.providers import OllamaProvider

    seen: list[dict] = []

    def transport(url, payload, timeout):
        seen.append(payload)
        return {
            "done": True,
            "done_reason": "stop",
            "message": {
                "content": '{"action":"get_balance","arguments":{"asset_id":"base:usdc"}}'
            },
        }

    provider = OllamaProvider(
        model="gemma4:e2b", transport=transport, skill="ROUTE RULES"
    )
    provider.propose_tool_call(
        {"user_request": "usdc balance"}, ["get_balance"]
    )

    # The routing skill must not appear in the argument-phase prompt.
    assert "ROUTE RULES" not in str(seen[-1]["messages"])


def test_argument_skill_reaches_only_the_argument_phase():
    from agentic_wallet.providers import OllamaProvider

    seen: list[dict] = []

    def transport(url, payload, timeout):
        seen.append(payload)
        return {
            "done": True,
            "done_reason": "stop",
            "message": {
                "content": '{"action":"get_balance","arguments":{"asset_id":"base:usdc"}}'
            },
        }

    provider = OllamaProvider(
        model="gemma4:e2b", transport=transport, argument_skill="ARG RULES"
    )
    provider.propose_tool_call({"user_request": "usdc balance"}, ["get_balance"])
    assert seen[-1]["messages"][0] == {"role": "system", "content": "ARG RULES"}

    # And the argument skill must not leak into routing.
    seen.clear()

    def route_transport(url, payload, timeout):
        seen.append(payload)
        return {
            "done": True,
            "done_reason": "stop",
            "message": {"content": '{"proposed_action":"get_account"}'},
        }

    provider = OllamaProvider(
        model="gemma4:e2b", transport=route_transport, argument_skill="ARG RULES"
    )
    provider.propose_dialogue_route(
        {"user_request": "my address"}, ["get_account"], ["get_account"]
    )
    assert "ARG RULES" not in str(seen[-1]["messages"])


def test_ollama_provider_without_skill_adds_no_system_message():
    from agentic_wallet.providers import OllamaProvider

    captured: dict = {}

    def transport(url, payload, timeout):
        captured["payload"] = payload
        return {
            "done": True,
            "done_reason": "stop",
            "message": {"content": '{"proposed_action":"get_account"}'},
        }

    provider = OllamaProvider(model="gemma4:e2b", transport=transport)
    provider.propose_dialogue_route(
        {"user_request": "what is my address?"},
        ["get_account", "get_portfolio"],
        ["get_portfolio"],
    )

    roles = [message["role"] for message in captured["payload"]["messages"]]
    assert roles.count("system") <= 1
    # Whatever system prompt exists is the contract's, not an injected skill.
    assert "ROUTING RULES" not in str(captured["payload"]["messages"])
