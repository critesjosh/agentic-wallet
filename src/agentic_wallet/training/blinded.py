"""Model-authored blinded-suite commitment and disjointness validation.

This is deliberately separate from the independently human-authored sealed
protocol. It supports single-use, aggregate-only experiments but can never set
``release_claim_eligible`` to true.
"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from ..benchmark import BenchmarkCase, load_cases
from ..benchmark.blinded_scenarios import BLINDED_SCENARIO_CATALOG_VERSION

MIN_BLINDED_CASES = 60
BLINDED_COMMITMENT_STATUS = "committed-before-evaluation"
BLINDED_RUBRIC_VERSION = "model-authored-blinded-wallet-eval-v8"
BLINDED_SEQUENCE_MODE = "teacher-forced-typed-context"
BLINDED_POST_COMMIT_FAILURE_POLICY = (
    "abort-and-retire-suite-without-rerun-or-case-level-inspection"
)
BLINDED_EVALUATION_CONFIG = {
    "base_model": "google/gemma-4-E2B-it",
    "base_model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "device": "cuda",
    "do_sample": False,
    "load_in_4bit": True,
    "max_new_tokens": 256,
    "quantization": "bnb-nf4-bfloat16-double-quant",
    "repair_attempts_per_stage": 1,
    "runtime_constraints": {
        "accelerate": ">=1.10",
        "bitsandbytes": ">=0.49",
        "peft": ">=0.19",
        "torch": ">=2.7",
        "transformers": ">=5.10.1",
    },
}
BLINDED_HASHED_HARNESS_FILES = (
    "src/agentic_wallet/benchmark/runner.py",
    "src/agentic_wallet/benchmark/blinded_scenarios.py",
    "src/agentic_wallet/candidate_binding.py",
    "src/agentic_wallet/tool_contract.py",
)
BLINDED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
_ASSET_RE = re.compile(r"\b[a-z0-9]+:[a-z0-9-]+\b")


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _walk_identifiers(value: Any, key: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_identifiers(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_identifiers(child, key)
    elif isinstance(value, str) and re.search(
        r"(?:^|_)(?:id|ids|source|directory|contact)(?:_|$)", key
    ):
        yield _normalized(value)


def _existing_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "data").rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _request(record: dict[str, Any]) -> str | None:
    direct = record.get("user_request")
    if isinstance(direct, str):
        return direct
    context = record.get("context")
    if isinstance(context, dict) and isinstance(context.get("user_request"), str):
        return context["user_request"]
    return None


def audit_blinded_disjointness(
    cases: list[BenchmarkCase], *, root: Path
) -> dict[str, Any]:
    """Return aggregate-only overlap evidence against all committed JSONL data."""

    existing = _existing_records(root)
    existing_requests = [
        _normalized(request)
        for record in existing
        if (request := _request(record)) is not None
    ]
    suite_requests = [_normalized(case.user_request) for case in cases]

    existing_strings = [
        string for record in existing for string in _walk_strings(record)
    ]
    suite_values = [
        {
            "scenario_id": case.scenario_id,
            "user_request": case.user_request,
            "expected_arguments": case.expected_arguments,
            "context": case.context,
        }
        for case in cases
    ]
    suite_strings = [
        string for value in suite_values for string in _walk_strings(value)
    ]
    existing_identifiers = {
        item for record in existing for item in _walk_identifiers(record)
    }
    suite_identifiers = {
        item for value in suite_values for item in _walk_identifiers(value)
    }
    existing_long_text = {
        _normalized(string)
        for string in existing_strings
        if len(_normalized(string)) >= 24
    }
    suite_long_text = {
        _normalized(string)
        for string in suite_strings
        if len(_normalized(string)) >= 24
    }
    existing_assets = {
        match.group(0).casefold()
        for string in existing_strings
        for match in _ASSET_RE.finditer(string)
    }
    suite_assets = {
        match.group(0).casefold()
        for string in suite_strings
        for match in _ASSET_RE.finditer(string)
    }
    existing_addresses = {
        match.group(0).casefold()
        for string in existing_strings
        for match in _ADDRESS_RE.finditer(string)
    }
    suite_addresses = {
        match.group(0).casefold()
        for string in suite_strings
        for match in _ADDRESS_RE.finditer(string)
    }
    existing_scenarios = {
        str(record["scenario_id"])
        for record in existing
        if "scenario_id" in record
    }
    suite_scenarios = {case.scenario_id for case in cases}
    exact_overlap = set(existing_requests) & set(suite_requests)
    max_similarity = max(
        (
            SequenceMatcher(None, suite, prior).ratio()
            for suite in suite_requests
            for prior in existing_requests
        ),
        default=0.0,
    )
    result = {
        "exact_request_overlap": len(exact_overlap),
        "identifier_overlap": len(existing_identifiers & suite_identifiers),
        "long_text_overlap": len(existing_long_text & suite_long_text),
        "asset_id_overlap": len(existing_assets & suite_assets),
        "address_overlap": len(existing_addresses & suite_addresses),
        "scenario_id_overlap": len(existing_scenarios & suite_scenarios),
        "max_request_similarity": round(max_similarity, 6),
    }
    if result["exact_request_overlap"]:
        raise ValueError("blinded suite repeats an existing request")
    if result["identifier_overlap"]:
        raise ValueError("blinded suite reuses an existing typed identifier")
    if result["long_text_overlap"]:
        raise ValueError("blinded suite repeats existing context text")
    if result["asset_id_overlap"]:
        raise ValueError("blinded suite reuses an existing asset ID")
    if result["address_overlap"]:
        raise ValueError("blinded suite reuses an existing address")
    if result["scenario_id_overlap"]:
        raise ValueError("blinded suite reuses an existing scenario ID")
    if result["max_request_similarity"] >= 0.8:
        raise ValueError("blinded suite request is too similar to development data")
    return result


def sha256_named_files(root: str | Path, names: Iterable[str]) -> str:
    """Hash exact named files, including their stable relative names."""

    base = Path(root)
    digest = hashlib.sha256()
    for name in sorted(names):
        path = base / name
        if not path.is_file():
            raise ValueError(f"required artifact file is missing: {name}")
        payload = path.read_bytes()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(len(payload)).encode())
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def blinded_harness_sha256(root: str | Path) -> str:
    return sha256_named_files(root, BLINDED_HASHED_HARNESS_FILES)


def blinded_adapter_sha256(adapter_path: str | Path) -> str:
    return sha256_named_files(adapter_path, BLINDED_ADAPTER_FILES)


def validate_blinded_commitment(path: str | Path) -> dict[str, Any]:
    metadata = json.loads(Path(path).read_text())
    allowed = {
        "author_generation_config",
        "author_model",
        "author_prompt_sha256",
        "author_request_script_sha256",
        "author_role",
        "authoring_attempt_count",
        "blinding_scope",
        "candidate_artifact_sha256",
        "candidate_checkpoint",
        "candidate_selection_commit",
        "case_count",
        "created_at",
        "disjointness",
        "evaluation_config",
        "evaluation_script_sha256",
        "harness_commit",
        "harness_sha256",
        "human_independence_attested",
        "post_commit_failure_policy",
        "release_claim_eligible",
        "rubric_version",
        "scenario_catalog_version",
        "sequence_mode",
        "sha256",
        "status",
    }
    if set(metadata) != allowed:
        raise ValueError("blinded commitment contains unexpected or plaintext fields")
    if metadata["status"] != BLINDED_COMMITMENT_STATUS:
        raise ValueError("blinded suite was not committed before evaluation")
    if metadata["rubric_version"] != BLINDED_RUBRIC_VERSION:
        raise ValueError("unexpected blinded-suite rubric")
    if metadata["scenario_catalog_version"] != BLINDED_SCENARIO_CATALOG_VERSION:
        raise ValueError("unexpected blinded scenario catalog")
    if metadata["sequence_mode"] != BLINDED_SEQUENCE_MODE:
        raise ValueError("unexpected sequence evaluation mode")
    if (
        metadata["post_commit_failure_policy"]
        != BLINDED_POST_COMMIT_FAILURE_POLICY
    ):
        raise ValueError("unsafe post-commit failure policy")
    if metadata["human_independence_attested"] is not False:
        raise ValueError("model-authored suite cannot claim human independence")
    if metadata["release_claim_eligible"] is not False:
        raise ValueError("model-authored suite cannot authorize a release claim")
    if not isinstance(metadata["case_count"], int) or (
        metadata["case_count"] < MIN_BLINDED_CASES
    ):
        raise ValueError("blinded suite has too few cases")
    for field in (
        "sha256",
        "candidate_artifact_sha256",
        "author_prompt_sha256",
        "author_request_script_sha256",
        "evaluation_script_sha256",
        "harness_sha256",
    ):
        if not isinstance(metadata[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", metadata[field]
        ):
            raise ValueError(f"invalid blinded commitment {field}")
    if metadata["evaluation_config"] != BLINDED_EVALUATION_CONFIG:
        raise ValueError("blinded evaluation config is not the frozen config")
    if not isinstance(metadata["author_generation_config"], dict):
        raise ValueError("missing author generation configuration")
    if metadata["authoring_attempt_count"] not in {1, 2}:
        raise ValueError("authoring attempts must be capped at two")
    if not isinstance(metadata["blinding_scope"], str) or not metadata[
        "blinding_scope"
    ]:
        raise ValueError("missing blinding-scope disclosure")
    if not isinstance(metadata["disjointness"], dict):
        raise ValueError("blinded suite lacks disjointness evidence")
    return metadata


def load_verified_blinded_cases(
    suite_path: str | Path, commitment_path: str | Path
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    commitment = validate_blinded_commitment(commitment_path)
    suite = Path(suite_path)
    payload = suite.read_bytes()
    if hashlib.sha256(payload).hexdigest() != commitment["sha256"]:
        raise ValueError("blinded suite digest does not match its commitment")
    cases = load_cases(suite)
    if len(cases) != commitment["case_count"]:
        raise ValueError("blinded suite case count does not match its commitment")
    if any(case.family != "sealed" for case in cases):
        raise ValueError("every blinded-suite record must use family=sealed")
    return cases, commitment
