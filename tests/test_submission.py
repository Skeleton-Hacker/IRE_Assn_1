from pathlib import Path

import polars as pl
import pytest

from ire_assn1.experiments.submission import (
    _default_submission_path,
    submit_from_config,
    write_competition_submission,
    write_mind_submission,
)


def test_mind_submission_uses_original_candidate_positions(tmp_path: Path) -> None:
    predictions = pl.DataFrame(
        {
            "impression_id": [
                "mind:small:impression:7",
                "mind:small:impression:7",
                "mind:small:impression:7",
            ],
            "article_id": ["a", "b", "c"],
            "position": [0, 1, 2],
            "score": [0.1, 0.9, 0.2],
        }
    )
    output = tmp_path / "mind.txt"
    write_mind_submission(predictions, output)
    assert output.read_text(encoding="utf-8") == "7 [3,1,2]\n"


def test_competition_submission_rejects_incomplete_positions(tmp_path: Path) -> None:
    predictions = pl.DataFrame(
        {
            "impression_id": ["ebnerd:large:impression:7"],
            "article_id": ["a"],
            "position": [1],
            "score": [0.1],
        }
    )
    with pytest.raises(ValueError, match="complete permutation"):
        write_competition_submission(predictions, tmp_path / "submission.txt")


def test_default_submission_paths_use_codabench_filenames() -> None:
    assert _default_submission_path("mind", "large", "bge").name == "prediction.txt"
    assert _default_submission_path("ebnerd", "large", "bm25").name == "predictions.txt"


def test_submit_from_config_writes_text_and_zip(tmp_path: Path) -> None:
    predictions = pl.DataFrame(
        {
            "impression_id": [
                "mind:large:impression:7",
                "mind:large:impression:7",
            ],
            "article_id": ["a", "b"],
            "position": [0, 1],
            "score": [0.1, 0.9],
        }
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions.write_parquet(predictions_path)
    expected_path = (
        tmp_path / "processed" / "mind" / "large" / "competition_test" / "impressions.parquet"
    )
    expected_path.parent.mkdir(parents=True)
    pl.DataFrame({"impression_id": ["mind:large:impression:7"]}).write_parquet(expected_path)
    output = tmp_path / "submission.txt"

    text_path, archive_path = submit_from_config(
        {
            "name": "mind",
            "variant": "large",
            "system": "bm25",
            "data": str(tmp_path),
            "predictions": str(predictions_path),
            "submission": str(output),
        }
    )

    assert text_path == output
    assert archive_path.is_file()
    assert output.read_text(encoding="utf-8") == "7 [2,1]\n"
