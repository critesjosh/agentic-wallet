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

BLINDED_CASE_COUNT = 64
BLINDED_COMMITMENT_STATUS = "committed-before-evaluation"
BLINDED_RUBRIC_VERSION = "model-authored-blinded-wallet-eval-v12"
BLINDED_SEQUENCE_MODE = "teacher-forced-typed-context"
BLINDED_POST_COMMIT_FAILURE_POLICY = (
    "abort-and-retire-suite-without-rerun-or-case-level-inspection"
)
BLINDED_AUTHOR_GENERATION_CONFIG = {
    "batch_count": 8,
    "filesystem_output": "external-only",
    "local_validation_attempts_per_shard": 3,
    "requested_fork_turns": "none",
    "interface": "codex-isolated-terra-subagent-with-frozen-validator",
    "model_alias": "gpt-5.6-terra",
    "requested_repository_access": "exact-validator-command-only",
    "whole_suite_regeneration_only": True,
}
BLINDED_AUTHOR_MODEL = "codex/gpt-5.6-terra"
BLINDED_AUTHOR_ROLE = "model-authored blinded evaluator"
BLINDED_BLINDING_SCOPE = (
    "Terra subagents were invoked without forked conversation and instructed "
    "not to access repository training or development plaintext; "
    "the evaluator receives no case-level output. The developer operates "
    "the workflow, so this is not independent-human evidence."
)
BLINDED_CANDIDATE_ARTIFACT_SHA256 = (
    "d1602c2f94835ef42113c7394a5918263ddc31860fee2ef0e3fdddb33d73abc9"
)
BLINDED_CANDIDATE_CHECKPOINT = "checkpoint-25"
BLINDED_CANDIDATE_SELECTION_COMMIT = "fc0547e"
BLINDED_EVALUATION_CONFIG = {
    "base_model": "google/gemma-4-E2B-it",
    "base_model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "cuda_version": "13.0",
    "custody_bucket": "hf://buckets/critesjosh/agentic-wallet-smoke",
    "custody_root": "/evaluation-custody",
    "device": "cuda",
    "do_sample": False,
    "load_in_4bit": True,
    "max_new_tokens": 256,
    "job_flavor": "l4x1",
    "job_image": "ghcr.io/astral-sh/uv:python3.12-bookworm",
    "pythonpath": "/workspace/src",
    "quantization": "bnb-nf4-bfloat16-double-quant",
    "repair_attempts_per_stage": 1,
    "python_version": "3.12.12",
    "source_root": "/workspace",
    "runtime_constraints": {
        "accelerate": "==1.14.0",
        "bitsandbytes": "==0.49.2",
        "eth-utils": "==6.0.0",
        "packaging": "==26.2",
        "peft": "==0.19.1",
        "pydantic": "==2.13.4",
        "torch": "==2.13.0",
        "torchvision": "==0.28.0",
        "transformers": "==5.10.1",
    },
}
BLINDED_HASHED_HARNESS_FILES = (
    "src/agentic_wallet/benchmark/cases.py",
    "src/agentic_wallet/benchmark/loader.py",
    "src/agentic_wallet/benchmark/runner.py",
    "src/agentic_wallet/benchmark/blinded_scenarios.py",
    "src/agentic_wallet/blinded_evaluation.py",
    "src/agentic_wallet/candidate_binding.py",
    "src/agentic_wallet/inference.py",
    "src/agentic_wallet/providers/local_transformers.py",
    "src/agentic_wallet/schemas/conversation.py",
    "src/agentic_wallet/schemas/common.py",
    "src/agentic_wallet/schemas/dialogue.py",
    "src/agentic_wallet/schemas/tool_call.py",
    "src/agentic_wallet/benchmark/registries.py",
    "src/agentic_wallet/tool_contract.py",
    "src/agentic_wallet/training/blinded.py",
    "src/agentic_wallet/training/blinded_authoring.py",
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
        r"(?:^|_)(?:"
        r"id|ids|source|directory|contact|recipient|recipients|"
        r"asset|assets|token|tokens|address|addresses"
        r")(?:_|$)",
        key.casefold(),
    ):
        yield _normalized(value)


def _walk_text(
    value: Any, key: str = "", *, inside_untrusted_data: bool = False
) -> Iterable[str]:
    """Yield exact text carried in request or explicitly untrusted fields."""

    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_text(
                child,
                str(child_key),
                inside_untrusted_data=inside_untrusted_data
                or str(child_key).casefold() == "untrusted_data",
            )
    elif isinstance(value, list):
        for child in value:
            yield from _walk_text(
                child, key, inside_untrusted_data=inside_untrusted_data
            )
    elif isinstance(value, str) and (
        inside_untrusted_data
        or re.search(
            r"(?:^|_)(?:user_request|request|text|memo|description|"
            r"message|name|reason|untrusted_data)(?:_|$)",
            key.casefold(),
        )
    ):
        normalized = _normalized(value)
        if normalized:
            yield normalized


def _plaintext_development_paths(root: Path) -> Iterable[Path]:
    """Yield committed development/training records, not derived aggregates."""

    data = root / "data"
    for directory in (data / "benchmark", data / "training"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix not in {".json", ".jsonl"} or not path.is_file():
                continue
            relative_parts = path.relative_to(directory).parts
            if "results" in relative_parts:
                continue
            if path.name.endswith((".commitment.json", ".manifest.json")):
                continue
            yield path


def _existing_records(root: Path) -> list[Any]:
    records: list[Any] = []
    for path in _plaintext_development_paths(root):
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
        else:
            records.append(json.loads(path.read_text()))
    return records


def _walk_requests(value: Any, key: str = "") -> Iterable[str]:
    """Yield user requests from arbitrarily nested JSON development sources."""

    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_requests(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_requests(child, key)
    elif isinstance(value, str) and key.casefold() == "user_request":
        yield value


def _walk_scenario_ids(value: Any, key: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_scenario_ids(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_scenario_ids(child, key)
    elif isinstance(value, str) and key.casefold() == "scenario_id":
        yield value


def audit_blinded_disjointness(
    cases: list[BenchmarkCase], *, root: Path
) -> dict[str, Any]:
    """Return aggregate-only overlap evidence against committed plaintext data."""

    existing = _existing_records(root)
    existing_requests = [
        _normalized(request)
        for record in existing
        for request in _walk_requests(record)
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
    existing_text = {text for record in existing for text in _walk_text(record)}
    suite_text = {text for value in suite_values for text in _walk_text(value)}
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
        scenario_id
        for record in existing
        for scenario_id in _walk_scenario_ids(record)
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
        "text_overlap": len(existing_text & suite_text),
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
    if result["text_overlap"]:
        raise ValueError("blinded suite repeats existing typed or untrusted text")
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
        "author_procedure_sha256",
        "author_shard_sha256",
        "author_validator_sha256",
        "author_role",
        "authoring_attempt_count",
        "blinding_scope",
        "candidate_artifact_sha256",
        "candidate_checkpoint",
        "candidate_selection_commit",
        "case_count",
        "commit_script_sha256",
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
    if metadata["author_model"] != BLINDED_AUTHOR_MODEL:
        raise ValueError("unexpected blinded-suite author model")
    if metadata["author_role"] != BLINDED_AUTHOR_ROLE:
        raise ValueError("unexpected blinded-suite author role")
    if metadata["blinding_scope"] != BLINDED_BLINDING_SCOPE:
        raise ValueError("unexpected blinded-suite blinding scope")
    if (
        metadata["candidate_artifact_sha256"]
        != BLINDED_CANDIDATE_ARTIFACT_SHA256
        or metadata["candidate_checkpoint"] != BLINDED_CANDIDATE_CHECKPOINT
        or metadata["candidate_selection_commit"]
        != BLINDED_CANDIDATE_SELECTION_COMMIT
    ):
        raise ValueError("unexpected frozen candidate identity")
    if (
        not isinstance(metadata["case_count"], int)
        or isinstance(metadata["case_count"], bool)
        or metadata["case_count"] != BLINDED_CASE_COUNT
    ):
        raise ValueError(
            f"blinded suite must contain exactly {BLINDED_CASE_COUNT} cases"
        )
    for field in (
        "sha256",
        "candidate_artifact_sha256",
        "author_prompt_sha256",
        "author_procedure_sha256",
        "author_validator_sha256",
        "commit_script_sha256",
        "evaluation_script_sha256",
        "harness_sha256",
    ):
        if not isinstance(metadata[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", metadata[field]
        ):
            raise ValueError(f"invalid blinded commitment {field}")
    if not isinstance(metadata["harness_commit"], str) or re.fullmatch(
        r"[0-9a-f]{40}", metadata["harness_commit"]
    ) is None:
        raise ValueError("invalid blinded commitment harness_commit")
    if metadata["evaluation_config"] != BLINDED_EVALUATION_CONFIG:
        raise ValueError("blinded evaluation config is not the frozen config")
    if metadata["author_generation_config"] != BLINDED_AUTHOR_GENERATION_CONFIG:
        raise ValueError("unexpected author generation configuration")
    shard_digests = metadata["author_shard_sha256"]
    if (
        not isinstance(shard_digests, list)
        or len(shard_digests) != BLINDED_AUTHOR_GENERATION_CONFIG["batch_count"]
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in shard_digests
        )
    ):
        raise ValueError("invalid author shard digests")
    if (
        not isinstance(metadata["authoring_attempt_count"], int)
        or isinstance(metadata["authoring_attempt_count"], bool)
        or metadata["authoring_attempt_count"] not in {1, 2}
    ):
        raise ValueError("authoring attempts must be capped at two")
    disjointness = metadata["disjointness"]
    expected_disjointness_keys = {
        "exact_request_overlap",
        "identifier_overlap",
        "text_overlap",
        "long_text_overlap",
        "asset_id_overlap",
        "address_overlap",
        "scenario_id_overlap",
        "max_request_similarity",
    }
    if not isinstance(disjointness, dict) or set(disjointness) != (
        expected_disjointness_keys
    ):
        raise ValueError("blinded suite lacks disjointness evidence")
    overlap_keys = expected_disjointness_keys - {"max_request_similarity"}
    if any(
        not isinstance(disjointness[key], int)
        or isinstance(disjointness[key], bool)
        or disjointness[key] != 0
        for key in overlap_keys
    ):
        raise ValueError("blinded suite has nonzero disjointness overlap")
    similarity = disjointness["max_request_similarity"]
    if (
        not isinstance(similarity, (int, float))
        or isinstance(similarity, bool)
        or similarity < 0
        or similarity >= 0.8
    ):
        raise ValueError("blinded suite has invalid request similarity evidence")
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
    if len(cases) != BLINDED_CASE_COUNT:
        raise ValueError(
            f"blinded suite must contain exactly {BLINDED_CASE_COUNT} cases"
        )
    if len(cases) != commitment["case_count"]:
        raise ValueError("blinded suite case count does not match its commitment")
    if any(case.family != "sealed" for case in cases):
        raise ValueError("every blinded-suite record must use family=sealed")
    return cases, commitment
