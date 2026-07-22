"""Load benchmark cases from JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .cases import BenchmarkCase


def load_cases(path: Union[str, Path]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(BenchmarkCase(**json.loads(line)))
    return cases
