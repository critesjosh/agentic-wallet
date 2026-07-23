"""Contract tests for V14 blinded language-seed expansion."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pytest

from agentic_wallet.benchmark.blinded_scenarios import compile_blinded_source
from agentic_wallet.training import blinded_seed_authoring
from agentic_wallet.training.blinded import BLINDED_CASE_COUNT
from agentic_wallet.training.blinded import BLINDED_HASHED_HARNESS_FILES


def _valid_shard(prefix: str) -> list[dict[str, object]]:
    remaining = Counter(
        blinded_seed_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]
    )
    trajectory = list(
        blinded_seed_authoring.EXPECTED_TRAJECTORY_SCENARIOS[prefix]
    )
    remaining.subtract(trajectory)
    scenario_types = trajectory + [
        scenario_type
        for scenario_type, count in remaining.items()
        for _ in range(count)
    ]
    shard_name = prefix.removesuffix("-")
    return [
        {
            "scenario_type": scenario_type,
            "utterance": f"Please handle deterministic sealed case {index} safely.",
            "world_seed": f"world-{shard_name}-{index}",
            "trajectory_key": f"trajectory-{shard_name}" if index < 4 else None,
            "turn_index": index if index < 4 else None,
        }
        for index, scenario_type in enumerate(scenario_types)
    ]


def _write_shard(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


@pytest.mark.parametrize("prefix", blinded_seed_authoring.EXPECTED_SHARD_PREFIXES)
def test_each_seed_shard_compiles_its_exact_scenario_quota(prefix: str):
    values = _valid_shard(prefix)

    cases, counts = blinded_seed_authoring._validate_seed_shard(values, prefix)

    assert len(cases) == 8
    assert counts == blinded_seed_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]


def test_every_catalogued_seed_scenario_type_compiles_to_a_sealed_case():
    compiled_by_type = {}
    for prefix in blinded_seed_authoring.EXPECTED_SHARD_PREFIXES:
        for index, seed in enumerate(_valid_shard(prefix)):
            source = blinded_seed_authoring._expand_seed(
                seed, prefix=prefix, index=index
            )
            compiled_by_type[seed["scenario_type"]] = compile_blinded_source(source)

    assert set(compiled_by_type) == set(
        blinded_seed_authoring.EXPECTED_SCENARIO_COUNTS
    )
    assert {case.family for case in compiled_by_type.values()} == {"sealed"}


def test_seed_expansion_is_deterministic_and_code_owned():
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    seed = _valid_shard(prefix)[0]

    first = blinded_seed_authoring._expand_seed(seed, prefix=prefix, index=0)
    second = blinded_seed_authoring._expand_seed(seed, prefix=prefix, index=0)

    assert first == second
    assert first["user_request"].startswith(seed["utterance"])
    assert first["context"]["canonical_asset_ids"]
    assert first["id"].startswith(prefix)
    assert first["scenario_id"].startswith(f"{prefix}scenario-")


def test_trajectory_turns_share_world_and_receive_teacher_forced_history():
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    cases, _ = blinded_seed_authoring._validate_seed_shard(
        _valid_shard(prefix), prefix
    )
    turns = cases[:4]

    assert len({case.context["conversation_world_id"] for case in turns}) == 1
    assert len(
        {tuple(case.context["canonical_asset_ids"]) for case in turns}
    ) == 1
    assert [
        len(case.context["conversation_ledger"]["recent_messages"])
        for case in turns
    ] == [0, 1, 2, 3]
    assert [
        case.context["conversation_ledger"]["workflow_state"] for case in turns
    ] == [case.workflow_state for case in turns]


def test_seed_shard_rejects_invalid_fields_and_world_seeds():
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    seed = _valid_shard(prefix)[0]

    with_extra_field = {**seed, "expected_action": "get_balance"}
    with pytest.raises(ValueError, match="invalid fields"):
        blinded_seed_authoring._expand_seed(
            with_extra_field, prefix=prefix, index=0
        )

    malformed_world = {**seed, "world_seed": "Uppercase-world"}
    with pytest.raises(ValueError, match="world seed"):
        blinded_seed_authoring._expand_seed(
            malformed_world, prefix=prefix, index=0
        )

    whitespace_utterance = {**seed, "utterance": "            "}
    with pytest.raises(ValueError, match="utterance"):
        blinded_seed_authoring._expand_seed(
            whitespace_utterance, prefix=prefix, index=0
        )


@pytest.mark.parametrize(
    ("trajectory_key", "turn_index"),
    [
        ("trajectory-valid", None),
        (None, 0),
        ("trajectory-valid", True),
        ("BadTrajectory", 0),
    ],
)
def test_seed_rejects_invalid_trajectory_metadata(
    trajectory_key: object, turn_index: object
):
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    seed = {
        **_valid_shard(prefix)[0],
        "trajectory_key": trajectory_key,
        "turn_index": turn_index,
    }

    with pytest.raises(ValueError, match="trajectory"):
        blinded_seed_authoring._expand_seed(seed, prefix=prefix, index=0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.__setitem__(0, {**values[0], "trajectory_key": None, "turn_index": None}),
        lambda values: values[3].__setitem__("turn_index", 4),
    ],
    ids=("wrong-trajectory-count", "noncontiguous-turns"),
)
def test_seed_shard_rejects_invalid_trajectory_shape(mutate):
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    values = _valid_shard(prefix)
    mutate(values)

    with pytest.raises(ValueError, match="trajectory"):
        blinded_seed_authoring._validate_seed_shard(values, prefix)


def test_seed_shard_rejects_out_of_order_trajectory_turns():
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    values = _valid_shard(prefix)
    values[0], values[1] = values[1], values[0]

    with pytest.raises(ValueError, match="trajectory turns"):
        blinded_seed_authoring._validate_seed_shard(values, prefix)


def test_seed_shard_rejects_duplicate_world_seed_and_quota_drift():
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    duplicate_worlds = _valid_shard(prefix)
    duplicate_worlds[1]["world_seed"] = duplicate_worlds[0]["world_seed"]
    with pytest.raises(ValueError, match="world seeds must be unique"):
        blinded_seed_authoring._validate_seed_shard(duplicate_worlds, prefix)

    quota_drift = _valid_shard(prefix)
    quota_drift[0]["scenario_type"] = "conceptual_help"
    with pytest.raises(ValueError, match="quota"):
        blinded_seed_authoring._validate_seed_shard(quota_drift, prefix)

    trajectory_overlap = _valid_shard(prefix)
    trajectory_overlap[4]["world_seed"] = trajectory_overlap[0]["trajectory_key"]
    with pytest.raises(ValueError, match="must not overlap"):
        blinded_seed_authoring._validate_seed_shard(trajectory_overlap, prefix)


def test_author_seed_validator_returns_aggregate_only_results():
    prefix = blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[0]
    valid_report = blinded_seed_authoring.author_seed_validation_report(
        _valid_shard(prefix), prefix
    )
    assert valid_report == {"case_count": 8, "valid": True}

    invalid = _valid_shard(prefix)
    invalid[0]["utterance"] = "secret"
    invalid_report = blinded_seed_authoring.author_seed_validation_report(
        invalid, prefix
    )
    assert invalid_report == {"valid": False}
    assert "secret" not in json.dumps(invalid_report)


def test_teacher_forced_ledger_does_not_resolve_ambiguous_or_conflicting_facts():
    ambiguous_prefix = "tb122b-"
    ambiguous_cases, _ = blinded_seed_authoring._validate_seed_shard(
        _valid_shard(ambiguous_prefix), ambiguous_prefix
    )
    ambiguous = ambiguous_cases[0]
    wrong_chain = ambiguous_cases[1]

    assert ambiguous.context["conversation_ledger"]["resolved_intent"][
        "asset_id"
    ] is None
    assert wrong_chain.context["conversation_ledger"]["resolved_intent"][
        "chain_id"
    ] is None

    read_prefix = "tb121a-"
    read_cases, _ = blinded_seed_authoring._validate_seed_shard(
        _valid_shard(read_prefix), read_prefix
    )
    portfolio = read_cases[0]
    assert portfolio.context["conversation_ledger"]["resolved_intent"][
        "asset_id"
    ] is None


@pytest.mark.parametrize(
    "scenario_type",
    [
        "simulation_mismatch",
        "duplicate_plan",
        "unlimited_approval_attack",
        "prompt_injection",
        "signing_boundary",
    ],
)
def test_code_owned_safety_prompts_do_not_reveal_gold_decision(
    scenario_type: str,
):
    prefix = next(
        shard_prefix
        for shard_prefix, counts in (
            blinded_seed_authoring.EXPECTED_SHARD_SCENARIO_COUNTS.items()
        )
        if scenario_type in counts
    )
    seed = next(
        seed
        for seed in _valid_shard(prefix)
        if seed["scenario_type"] == scenario_type
    )
    seed["utterance"] = "Please help me understand what should happen next."

    request = blinded_seed_authoring._expand_seed(
        seed, prefix=prefix, index=0
    )["user_request"].casefold()

    assert "reject" not in request
    assert "do not sign" not in request
    assert "do not duplicate" not in request
    assert "unexpected debit" not in request
    assert "outside the supported" not in request


def test_materialization_compiles_exactly_64_unique_cases_and_global_trajectory_shape(
    tmp_path: Path,
):
    paths = []
    for prefix in blinded_seed_authoring.EXPECTED_SHARD_PREFIXES:
        path = tmp_path / f"{prefix}seeds.jsonl"
        _write_shard(path, _valid_shard(prefix))
        paths.append(path)

    cases, manifest = blinded_seed_authoring.materialize_author_seed_shards(paths)

    assert len(cases) == BLINDED_CASE_COUNT == 64
    assert len({case.id for case in cases}) == BLINDED_CASE_COUNT
    assert len({case.scenario_id for case in cases}) == BLINDED_CASE_COUNT
    assert manifest["scenario_counts"] == dict(
        sorted(blinded_seed_authoring.EXPECTED_SCENARIO_COUNTS.items())
    )
    assert manifest["source_sha256"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ]

    trajectories: defaultdict[str, list[int]] = defaultdict(list)
    independent = 0
    for case in cases:
        if case.trajectory_id is None:
            independent += 1
        else:
            trajectories[case.trajectory_id].append(case.turn_index)
    assert independent == 32
    assert len(trajectories) == 8
    assert set(map(tuple, trajectories.values())) == {tuple(range(4))}
    assert manifest["trajectory_count"] == 8


def test_materialization_requires_all_eight_shards(tmp_path: Path):
    paths = []
    for prefix in blinded_seed_authoring.EXPECTED_SHARD_PREFIXES[:-1]:
        path = tmp_path / f"{prefix}seeds.jsonl"
        _write_shard(path, _valid_shard(prefix))
        paths.append(path)

    with pytest.raises(ValueError, match="exactly eight"):
        blinded_seed_authoring.materialize_author_seed_shards(paths)


def test_materialization_rejects_world_seed_reused_across_shards(tmp_path: Path):
    paths = []
    duplicate = "globally-duplicate-world"
    for shard_index, prefix in enumerate(
        blinded_seed_authoring.EXPECTED_SHARD_PREFIXES
    ):
        values = _valid_shard(prefix)
        if shard_index < 2:
            values[0]["world_seed"] = duplicate
        path = tmp_path / f"{prefix}seeds.jsonl"
        _write_shard(path, values)
        paths.append(path)

    with pytest.raises(ValueError, match="globally unique"):
        blinded_seed_authoring.materialize_author_seed_shards(paths)


def test_seed_compiler_and_sequence_runner_are_frozen_harness_inputs():
    assert (
        "src/agentic_wallet/training/blinded_seed_authoring.py"
        in BLINDED_HASHED_HARNESS_FILES
    )
    assert (
        "src/agentic_wallet/benchmark/runner.py"
        in BLINDED_HASHED_HARNESS_FILES
    )


def test_v5_shard_prompts_match_frozen_quotas_and_trajectory_sequences():
    root = Path(__file__).resolve().parents[1]
    names = ("1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b")
    for prefix, name in zip(
        blinded_seed_authoring.EXPECTED_SHARD_PREFIXES, names, strict=True
    ):
        values = dict(
            line.split("=", 1)
            for line in (
                root / "docs" / f"terra-blinded-author-shard-{name}-v5.md"
            )
            .read_text()
            .splitlines()
            if "=" in line
        )
        assert values["prefix"] == prefix
        assert Counter(json.loads(values["scenario_counts"])) == (
            blinded_seed_authoring.EXPECTED_SHARD_SCENARIO_COUNTS[prefix]
        )
        assert tuple(json.loads(values["trajectory_scenarios"])) == (
            blinded_seed_authoring.EXPECTED_TRAJECTORY_SCENARIOS[prefix]
        )
