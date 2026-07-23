from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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

    assert cases["items"] == {"type": "string"}
    assert module.SHARD_SIZE == 8
    assert "expected_action" not in module.TOP_LEVEL_FIELDS
    assert "context" in module.TOP_LEVEL_FIELDS


def test_claude_request_rejects_checkout_output_before_invoking_claude(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    invoked = False

    def run(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("Claude Code must not be invoked")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "request_claude_blinded_shard.py",
            "--shared-prompt",
            "/does-not-need-to-exist",
            "--shard-prompt",
            "/does-not-need-to-exist",
            "--output",
            str(module.ROOT / "leaked-blinded-shard.jsonl"),
        ],
    )

    with pytest.raises(SystemExit, match="outside checkout"):
        module.main()

    assert not invoked


def test_claude_request_rejects_nonfrozen_prompt_before_invoking_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    invoked = False

    def run(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("Claude Code must not be invoked")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "request_claude_blinded_shard.py",
            "--shared-prompt",
            str(tmp_path / "not-frozen.md"),
            "--shard-prompt",
            str(next(iter(module.SHARD_PROMPTS))),
            "--output",
            str(tmp_path / "shard.jsonl"),
        ],
    )

    with pytest.raises(SystemExit, match="frozen author packet"):
        module.main()

    assert not invoked
