from __future__ import annotations

import json
from pathlib import Path

from ire_assn1.evaluation.types import EvaluationResult


def write_evaluation_result(result: EvaluationResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def read_evaluation_result(path: str | Path) -> EvaluationResult:
    return EvaluationResult.model_validate_json(Path(path).read_text(encoding="utf-8"))
