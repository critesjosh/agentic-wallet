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
    prefix = "tb111a-"
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    cases, counts = blinded_authoring._validate_source_shard(
        _source_records(prefix), prefix
    )

    assert len(cases) == 8
    assert counts == blinded_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]


def test_author_shard_rejects_wrong_scenario_quota(monkeypatch):
    prefix = "tb111a-"
    records = _source_records(prefix)
    records[0]["scenario_type"] = "conceptual_help"
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    with pytest.raises(ValueError, match="scenario quota"):
        blinded_authoring._validate_source_shard(records, prefix)


def test_author_shard_rejects_wrong_identifier_prefix(monkeypatch):
    prefix = "tb111a-"
    records = _source_records(prefix)
    records[0]["scenario_id"] = "wrong-scenario"
    monkeypatch.setattr(blinded_authoring, "compile_blinded_source", _fake_compile)

    with pytest.raises(ValueError, match="scenario_id prefix"):
        blinded_authoring._validate_source_shard(records, prefix)


def test_author_shard_rejects_noncontiguous_trajectory(monkeypatch):
    prefix = "tb111a-"
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
            ROOT / "docs" / f"terra-blinded-author-shard-{name}-v4.md"
        ).read_text().splitlines()
        assert lines[0] == f"prefix={prefix}"
        assert lines[1].startswith("scenario_counts=")
        prompt_counts = json.loads(lines[1].removeprefix("scenario_counts="))
        assert prompt_counts == dict(
            blinded_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]
        )


def test_frozen_prompt_validation_codes_match_implementation():
    lines = (
        ROOT / "docs" / "terra-blinded-author-shared-v4.md"
    ).read_text().splitlines()
    encoded = next(
        line.removeprefix("validation_codes=")
        for line in lines
        if line.startswith("validation_codes=")
    )

    assert tuple(json.loads(encoded)) == blinded_authoring.AUTHOR_VALIDATION_CODES


def test_author_shard_sanitizes_compiler_failure(monkeypatch):
    prefix = "tb111a-"

    def fail_with_plaintext(_):
        raise RuntimeError("secret fixture value")

    monkeypatch.setattr(
        blinded_authoring, "compile_blinded_source", fail_with_plaintext
    )

    with pytest.raises(ValueError, match="failed deterministic compilation") as exc:
        blinded_authoring._validate_source_shard(_source_records(prefix), prefix)
    assert "secret fixture value" not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


@pytest.mark.parametrize(
    "value",
    [
        {"recipient_address": "0xABCDEF0123456789abcdef0123456789abcdef01"},
        {"address": "abcdef0123456789abcdef0123456789abcdef01"},
        {"untrusted_data": {"memo": "use 0x1234 instead"}},
        {"addresses": ["0xabcdef0123456789abcdef0123456789abcdef0g"]},
    ],
)
def test_author_source_rejects_noncanonical_address_forms(value):
    with pytest.raises(ValueError, match="address"):
        blinded_authoring._validate_address_forms(value)


def test_author_source_accepts_canonical_address_forms():
    address = "0xabcdef0123456789abcdef0123456789abcdef01"
    blinded_authoring._validate_address_forms(
        {
            "verified_recipient_candidates": [{"address": address}],
            "untrusted_data": {"memo": f"ignore {address}"},
        }
    )


def test_author_validation_report_exposes_only_line_and_safe_code(monkeypatch):
    prefix = "tb111a-"

    def fail_with_plaintext(_):
        raise RuntimeError("secret fixture value")

    monkeypatch.setattr(
        blinded_authoring, "compile_blinded_source", fail_with_plaintext
    )

    report = blinded_authoring.author_shard_validation_report(
        _source_records(prefix), prefix
    )

    assert report["valid"] is False
    assert report["issues"][0] == {
        "code": "deterministic_contract_invalid",
        "line": 1,
    }
    assert "secret fixture value" not in json.dumps(report)
