"""Frozen v9 author-shard shape and deterministic suite materialization."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..benchmark import BenchmarkCase
from ..benchmark.blinded_scenarios import (
    benchmark_case_dict,
    compile_blinded_source,
)
from .blinded import BLINDED_CASE_COUNT

EXPECTED_TRAJECTORY_TURNS = frozenset(range(4))
EXPECTED_SHARD_SCENARIO_COUNTS: dict[str, Counter[str]] = {
    "tb91a-": Counter(
        {
            "read_portfolio": 2,
            "read_balance": 2,
            "read_allowances": 2,
            "read_registry": 2,
        }
    ),
    "tb91b-": Counter(
        {
            "conceptual_help": 3,
            "unsupported_request": 2,
            "transfer_complete": 3,
        }
    ),
    "tb92a-": Counter(
        {
            "transfer_complete": 1,
            "transfer_missing": 3,
            "transfer_untrusted_directory": 3,
            "transfer_wrong_chain": 1,
        }
    ),
    "tb92b-": Counter(
        {
            "transfer_wrong_chain": 1,
            "transfer_ambiguous_asset": 2,
            "transfer_missing_recipient": 2,
            "swap_quote": 3,
        }
    ),
    "tb93a-": Counter(
        {
            "swap_quote": 1,
            "quote_expired": 3,
            "simulation_mismatch": 4,
        }
    ),
    "tb93b-": Counter(
        {
            "simulation_match": 3,
            "cancel_workflow": 3,
            "duplicate_plan": 2,
        }
    ),
    "tb94a-": Counter(
        {
            "duplicate_plan": 1,
            "stale_portfolio": 3,
            "exact_approval": 3,
            "unlimited_approval_attack": 1,
        }
    ),
    "tb94b-": Counter(
        {
            "unlimited_approval_attack": 2,
            "prompt_injection": 3,
            "signing_boundary": 3,
        }
    ),
}
EXPECTED_SHARD_PREFIXES = tuple(EXPECTED_SHARD_SCENARIO_COUNTS)
EXPECTED_SCENARIO_COUNTS = sum(
    EXPECTED_SHARD_SCENARIO_COUNTS.values(), Counter()
)
_CANONICAL_ADDRESS_RE = re.compile(r"0x[0-9a-f]{40}")
_ADDRESS_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])0x[A-Za-z0-9]+")
_ADDRESS_KEY_RE = re.compile(r"(?:^|_)(?:address|addresses)(?:_|$)")


def _validate_address_forms(value: Any, key: str = "") -> None:
    """Require canonical lowercase address forms everywhere an author can place one."""

    if isinstance(value, dict):
        for child_key, child in value.items():
            _validate_address_forms(child, str(child_key).casefold())
        return
    if isinstance(value, list):
        for child in value:
            _validate_address_forms(child, key)
        return
    if not isinstance(value, str):
        return
    if _ADDRESS_KEY_RE.search(key) and _CANONICAL_ADDRESS_RE.fullmatch(value) is None:
        raise ValueError("address-valued source field is not canonical")
    for token in _ADDRESS_TOKEN_RE.findall(value):
        if _CANONICAL_ADDRESS_RE.fullmatch(token) is None:
            raise ValueError("address token is not canonical")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid author shard JSON on line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError("author shard records must be JSON objects")
        values.append(value)
    return values


def _validate_source_shard(
    values: list[dict[str, Any]], prefix: str
) -> tuple[list[BenchmarkCase], Counter[str]]:
    if len(values) != 8:
        raise ValueError("each author shard must contain exactly eight records")
    counts = Counter(value.get("scenario_type") for value in values)
    if counts != EXPECTED_SHARD_SCENARIO_COUNTS[prefix]:
        raise ValueError(f"author shard {prefix} has an invalid scenario quota")

    cases: list[BenchmarkCase] = []
    for value in values:
        _validate_address_forms(value)
        for field in ("id", "scenario_id"):
            identifier = value.get(field)
            if not isinstance(identifier, str) or not identifier.startswith(prefix):
                raise ValueError(
                    f"author shard {prefix} has an invalid {field} prefix"
                )
        compiled: BenchmarkCase | None
        try:
            compiled = compile_blinded_source(value)
        except Exception:
            compiled = None
        if compiled is None:
            raise ValueError(
                f"author shard {prefix} failed deterministic compilation"
            )
        cases.append(compiled)

    trajectories: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        if case.trajectory_id is None and case.turn_index is None:
            continue
        if (
            not isinstance(case.trajectory_id, str)
            or not case.trajectory_id.startswith(prefix)
            or not isinstance(case.turn_index, int)
            or isinstance(case.turn_index, bool)
        ):
            raise ValueError(f"author shard {prefix} has invalid trajectory metadata")
        trajectories.setdefault(case.trajectory_id, []).append(case)
    if len(trajectories) != 1:
        raise ValueError(f"author shard {prefix} must contain one trajectory")
    turns = next(iter(trajectories.values()))
    if (
        len(turns) != len(EXPECTED_TRAJECTORY_TURNS)
        or {case.turn_index for case in turns} != EXPECTED_TRAJECTORY_TURNS
    ):
        raise ValueError(
            f"author shard {prefix} trajectory must contain turns zero through three"
        )
    return cases, counts


def materialize_author_shards(
    source_paths: list[Path],
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    """Compile exactly eight isolated shards in their frozen canonical order."""

    if len(source_paths) != len(EXPECTED_SHARD_PREFIXES):
        raise ValueError("exactly eight author shard paths are required")
    cases: list[BenchmarkCase] = []
    source_sha256: list[str] = []
    scenario_counts: Counter[str] = Counter()
    seen_case_ids: set[str] = set()
    seen_scenario_ids: set[str] = set()
    for prefix, path in zip(EXPECTED_SHARD_PREFIXES, source_paths, strict=True):
        payload = path.read_bytes()
        shard_cases, shard_counts = _validate_source_shard(
            _read_jsonl(path), prefix
        )
        for case in shard_cases:
            if case.id in seen_case_ids or case.scenario_id in seen_scenario_ids:
                raise ValueError("author case and scenario IDs must be globally unique")
            seen_case_ids.add(case.id)
            seen_scenario_ids.add(case.scenario_id)
        cases.extend(shard_cases)
        scenario_counts.update(shard_counts)
        source_sha256.append(hashlib.sha256(payload).hexdigest())

    if len(cases) != BLINDED_CASE_COUNT:
        raise ValueError(
            f"blinded suite must contain exactly {BLINDED_CASE_COUNT} cases"
        )
    if scenario_counts != EXPECTED_SCENARIO_COUNTS:
        raise ValueError("blinded suite has an invalid aggregate scenario quota")
    return cases, {
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "source_sha256": source_sha256,
        "trajectory_count": len(EXPECTED_SHARD_PREFIXES),
    }


def canonical_suite_bytes(cases: list[BenchmarkCase]) -> bytes:
    return (
        "".join(
            json.dumps(benchmark_case_dict(case), sort_keys=True) + "\n"
            for case in cases
        )
    ).encode()
