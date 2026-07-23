from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "validate_blinded_author_shard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_blinded_author_shard", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_emits_only_aggregate_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    module = _module()
    source = tmp_path / "shard.jsonl"
    source.write_text("{}\n")
    monkeypatch.setattr(module, "_read_jsonl", lambda _: [{}] * 8)
    monkeypatch.setattr(
        module,
        "author_shard_validation_report",
        lambda *_: {"issues": [], "valid": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_blinded_author_shard.py",
            "--source",
            str(source),
            "--prefix",
            "tb111a-",
        ],
    )

    module.main()

    assert json.loads(capsys.readouterr().out) == {
        "case_count": 8,
        "issues": [],
        "valid": True,
    }


def test_validator_suppresses_failure_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    module = _module()
    source = tmp_path / "shard.jsonl"
    source.write_text("{}\n")
    monkeypatch.setattr(module, "_read_jsonl", lambda _: [{}] * 8)

    def fail(*_):
        raise RuntimeError("secret fixture")

    monkeypatch.setattr(module, "author_shard_validation_report", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_blinded_author_shard.py",
            "--source",
            str(source),
            "--prefix",
            "tb111a-",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "issues": [{"code": "source_json_invalid"}],
        "valid": False,
    }
