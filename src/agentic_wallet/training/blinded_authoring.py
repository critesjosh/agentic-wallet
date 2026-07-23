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
    "tb111a-": Counter(
        {
            "read_portfolio": 2,
            "read_balance": 2,
            "read_allowances": 2,
            "read_registry": 2,
        }
    ),
    "tb111b-": Counter(
        {
            "conceptual_help": 3,
            "unsupported_request": 2,
            "transfer_complete": 3,
        }
    ),
    "tb112a-": Counter(
        {
            "transfer_complete": 1,
            "transfer_missing": 3,
            "transfer_untrusted_directory": 3,
            "transfer_wrong_chain": 1,
        }
    ),
    "tb112b-": Counter(
        {
            "transfer_wrong_chain": 1,
            "transfer_ambiguous_asset": 2,
            "transfer_missing_recipient": 2,
            "swap_quote": 3,
        }
    ),
    "tb113a-": Counter(
        {
            "swap_quote": 1,
            "quote_expired": 3,
            "simulation_mismatch": 4,
        }
    ),
    "tb113b-": Counter(
        {
            "simulation_match": 3,
            "cancel_workflow": 3,
            "duplicate_plan": 2,
        }
    ),
    "tb114a-": Counter(
        {
            "duplicate_plan": 1,
            "stale_portfolio": 3,
            "exact_approval": 3,
            "unlimited_approval_attack": 1,
        }
    ),
    "tb114b-": Counter(
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
SOURCE_JSON_INVALID_CODE = "source_json_invalid"
AUTHOR_VALIDATION_CODES = (
    "address_form_invalid",
    "asset_not_in_canonical_assets",
    "canonical_assets_invalid",
    "complete_transfer_missing_trusted_fact",
    "deterministic_contract_invalid",
    "identifier_prefix_invalid",
    "incomplete_transfer_fixture_invalid",
    "incomplete_transfer_has_all_trusted_facts",
    "recipient_candidate_invalid",
    "record_contract_invalid",
    "record_count_invalid",
    "required_context_field_missing",
    "scenario_quota_invalid",
    "scenario_type_invalid",
    SOURCE_JSON_INVALID_CODE,
    "trajectory_or_shard_shape_invalid",
)


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


def _safe_compiler_error_code(exc: Exception) -> str:
    """Map compiler failures to value-free author repair categories."""

    message = str(exc)
    if "complete transfer did not produce" in message:
        return "complete_transfer_missing_trusted_fact"
    if "incomplete transfer unexpectedly has all" in message:
        return "incomplete_transfer_has_all_trusted_facts"
    if "incomplete transfer did not derive" in message:
        return "incomplete_transfer_fixture_invalid"
    if "missing deterministic argument fixtures" in message:
        return "required_context_field_missing"
    if "expected asset argument is not" in message:
        return "asset_not_in_canonical_assets"
    if "recipient candidate" in message:
        return "recipient_candidate_invalid"
    if "scenario source" in message or "scenario identifiers" in message:
        return "record_contract_invalid"
    if "canonical_asset_ids" in message:
        return "canonical_assets_invalid"
    if "unknown blinded scenario" in message:
        return "scenario_type_invalid"
    return "deterministic_contract_invalid"


def author_shard_validation_report(
    values: list[dict[str, Any]], prefix: str
) -> dict[str, Any]:
    """Return only line numbers and value-free repair codes to the shard author."""

    issues: list[dict[str, Any]] = []
    if len(values) != 8:
        return {
            "issues": [{"code": "record_count_invalid"}],
            "valid": False,
        }
    counts = Counter(value.get("scenario_type") for value in values)
    if counts != EXPECTED_SHARD_SCENARIO_COUNTS[prefix]:
        issues.append({"code": "scenario_quota_invalid"})
    for line_number, value in enumerate(values, 1):
        try:
            _validate_address_forms(value)
            for field in ("id", "scenario_id"):
                identifier = value.get(field)
                if (
                    not isinstance(identifier, str)
                    or not identifier.startswith(prefix)
                ):
                    raise ValueError(f"{field} prefix is invalid")
            case = compile_blinded_source(value)
        except Exception as exc:
            code = (
                "address_form_invalid"
                if "address" in str(exc).casefold()
                else "identifier_prefix_invalid"
                if "prefix is invalid" in str(exc)
                else _safe_compiler_error_code(exc)
            )
            issues.append({"code": code, "line": line_number})
        else:
            del case
    if not issues:
        try:
            _validate_source_shard(values, prefix)
        except Exception:
            issues.append({"code": "trajectory_or_shard_shape_invalid"})
    return {"issues": issues, "valid": not issues}


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
