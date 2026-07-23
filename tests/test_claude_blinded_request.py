from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "request_claude_blinded_shard.py"
    )
    spec = importlib.util.spec_from_file_location("claude_blinded_request", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claude_request_schema_requires_exact_strict_shard():
    module = _module()
    schema = module._schema()
    cases = schema["properties"]["cases"]
    item = cases["items"]

    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(module.TOP_LEVEL_FIELDS)
    assert "expected_action" not in item["properties"]
    assert item["properties"]["context_json"] == {"type": "string"}
