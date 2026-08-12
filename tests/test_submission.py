from pathlib import Path

import polars as pl

from ire_assn1.experiments.submission import write_mind_submission


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
