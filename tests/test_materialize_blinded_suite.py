from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_wallet.training import blinded_authoring

ROOT = Path(__file__).resolve().parents[1]


def _source_records(prefix: str) -> list[dict]:
    scenarios = [
        scenario
        for scenario, count in (
            blinded_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix].items()
        )
        for _ in range(count)
    ]
    return [
        {
            "id": f"{prefix}case-{index}",
            "scenario_id": f"{prefix}scenario-{index}",
            "scenario_type": scenario,
            "trajectory_id": f"{prefix}trajectory" if index < 4 else None,
            "turn_index": index if index < 4 else None,
        }
        for index, scenario in enumerate(scenarios)
    ]


def _fake_compile(value: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=value["id"],
        scenario_id=value["scenario_id"],
        trajectory_id=value["trajectory_id"],
        turn_index=value["turn_index"],
    )


def test_author_shard_requires_exact_quota_prefix_and_trajectory(monkeypatch):
    prefix = "tb81a-"
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    cases, counts = blinded_authoring._validate_source_shard(
        _source_records(prefix), prefix
    )

    assert len(cases) == 8
    assert counts == blinded_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]


def test_author_shard_rejects_wrong_scenario_quota(monkeypatch):
    prefix = "tb81a-"
    records = _source_records(prefix)
    records[0]["scenario_type"] = "conceptual_help"
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    with pytest.raises(ValueError, match="scenario quota"):
        blinded_authoring._validate_source_shard(records, prefix)


def test_author_shard_rejects_wrong_identifier_prefix(monkeypatch):
    prefix = "tb81a-"
    records = _source_records(prefix)
    records[0]["scenario_id"] = "wrong-scenario"
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    with pytest.raises(ValueError, match="scenario_id prefix"):
        blinded_authoring._validate_source_shard(records, prefix)


def test_author_shard_rejects_noncontiguous_trajectory(monkeypatch):
    prefix = "tb81a-"
    records = _source_records(prefix)
    records[3]["turn_index"] = 4
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    with pytest.raises(ValueError, match="turns zero through three"):
        blinded_authoring._validate_source_shard(records, prefix)


def test_all_frozen_terra_prompt_quotas_match_compiler_table():
    names = ("1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b")

    for prefix, name in zip(
        blinded_authoring.EXPECTED_SHARD_PREFIXES, names, strict=True
    ):
        lines = (
            ROOT / "docs" / f"terra-blinded-author-shard-{name}-v1.md"
        ).read_text().splitlines()
        assert lines[0] == f"prefix={prefix}"
        assert lines[1].startswith("scenario_counts=")
        prompt_counts = json.loads(lines[1].removeprefix("scenario_counts="))
        assert prompt_counts == dict(
            blinded_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]
        )
