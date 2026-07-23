from __future__ import annotations

import runpy
from pathlib import Path


def test_v5_checkpoint_eval_driver_is_evaluation_only(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    class _Completed:
        stdout = "{}\n"
        stderr = ""
        returncode = 0

        @staticmethod
        def check_returncode():
            return None

    monkeypatch.setenv("AGENTIC_WALLET_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kwargs: calls.append(command) or _Completed(),
    )

    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_hf_v5_checkpoint_eval.py"
        ),
        run_name="checkpoint_eval_test",
    )
    namespace["main"]()

    assert len(calls) == 8
    assert all("evaluate_development.py" in call[1] for call in calls)
    assert all("train_qlora.py" not in " ".join(call) for call in calls)
