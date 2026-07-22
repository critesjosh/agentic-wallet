import sys

import pytest

from agentic_wallet.inference import InferenceError
from agentic_wallet.providers.local_transformers import (
    LocalTransformersProvider,
    _extract_json,
)


def test_importing_provider_does_not_import_ml_dependencies():
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_extract_json_requires_the_whole_output():
    output = '{"action":"get_portfolio","arguments":{},"reason":"safe"}'
    assert _extract_json(output)["action"] == "get_portfolio"

    with pytest.raises(InferenceError, match="exactly one"):
        _extract_json(f"Result: {output}")


def test_extract_json_handles_braces_inside_strings():
    output = '{"action":"reject_request","arguments":{},"reason":"bad {input}"}'
    assert _extract_json(output)["reason"] == "bad {input}"


@pytest.mark.parametrize("output", ["no object", "[1, 2]", "{not-json}", '{} {}'])
def test_extract_json_fails_closed(output):
    with pytest.raises(InferenceError):
        _extract_json(output)


class _GeneratedProvider(LocalTransformersProvider):
    def _build_prompt(self, context, available_actions):
        return "prompt"

    def _generate(self, prompt):
        return self.output


def test_generated_disallowed_action_fails_closed():
    provider = _GeneratedProvider()
    provider.output = '{"action":"sign_transaction","arguments":{},"reason":"go"}'
    with pytest.raises(InferenceError, match="not available"):
        provider.propose_tool_call({}, ["read_portfolio"])


def test_generated_schema_violation_fails_closed():
    provider = _GeneratedProvider()
    provider.output = '{"action":7,"arguments":{},"reason":"go"}'
    with pytest.raises(InferenceError, match="invalid tool-call schema"):
        provider.propose_tool_call({}, ["read_portfolio"])


def test_prompt_falls_back_when_checkpoint_has_no_chat_template():
    provider = LocalTransformersProvider()
    provider._processor = type("Processor", (), {"chat_template": None})()
    prompt = provider._build_prompt(
        {"user_request": "show balances"}, ["get_portfolio"]
    )
    assert '"available_actions":["get_portfolio"]' in prompt
    assert prompt.endswith("JSON tool call:")
