from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "commit_blinded_suite.py"
    )
    spec = importlib.util.spec_from_file_location("commit_blinded_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_commitment_hashes_suite_not_prompt_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    suite_payload = b'{"suite":"payload"}\n'
    suite = tmp_path / "suite.jsonl"
    suite.write_bytes(suite_payload)
    sources = [tmp_path / f"source-{index}.jsonl" for index in range(8)]
    for path in sources:
        path.write_text("{}\n")
    output = tmp_path / "commitment.json"
    cases = [
        SimpleNamespace(family="sealed", hard_zero_category=None)
        for _ in range(64)
    ]
    receipt = {
        "source_sha256": [f"{index:064x}" for index in range(8)],
    }

    monkeypatch.setattr(
        module, "materialize_author_seed_shards", lambda _: (cases, receipt)
    )
    monkeypatch.setattr(module, "canonical_suite_bytes", lambda _: suite_payload)
    monkeypatch.setattr(module, "audit_blinded_disjointness", lambda *_args, **_kwargs: {
        "address_overlap": 0,
        "asset_id_overlap": 0,
        "exact_request_overlap": 0,
        "identifier_overlap": 0,
        "long_text_overlap": 0,
        "max_request_similarity": 0.1,
        "scenario_id_overlap": 0,
        "text_overlap": 0,
    })
    monkeypatch.setattr(module, "_require_frozen_git_state", lambda: "f" * 40)
    monkeypatch.setattr(module, "_prompt_digest", lambda: "a" * 64)
    monkeypatch.setattr(module, "sha256_named_files", lambda *_: "b" * 64)
    monkeypatch.setattr(module, "blinded_harness_sha256", lambda _: "c" * 64)
    argv = [
        "commit_blinded_suite.py",
        "--suite",
        str(suite),
        "--authoring-attempt-count",
        "1",
        "--output",
        str(output),
        "--acknowledge-model-authored",
    ]
    for source in sources:
        argv.extend(("--source", str(source)))
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    commitment = json.loads(output.read_text())
    assert commitment["sha256"] == hashlib.sha256(suite_payload).hexdigest()
    assert commitment["sha256"] != commitment["author_prompt_sha256"]


def test_commitment_output_is_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    output = tmp_path / "commitment.json"
    module._write_new_json(output, {"first": True})

    with pytest.raises(SystemExit, match="already exists"):
        module._write_new_json(output, {"first": False})
