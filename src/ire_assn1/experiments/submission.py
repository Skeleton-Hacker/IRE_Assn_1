from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import pyarrow.parquet as pq

from ire_assn1.paths import project_path
from ire_assn1.settings import load_mapping


def write_competition_submission(predictions: pl.DataFrame, output: Path) -> int:
    required = {"impression_id", "article_id", "position", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Competition predictions are missing columns: {missing_columns}")
    ranked = predictions.sort(
        ["impression_id", "score", "article_id", "position"],
        descending=[False, True, False, False],
    )
    grouped = ranked.group_by("impression_id", maintain_order=True).agg(
        pl.col("position"),
        pl.col("article_id"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for impression_id, positions, article_ids in grouped.iter_rows():
            complete = set(positions) == set(range(len(positions)))
            if len(set(positions)) != len(positions) or not complete:
                raise ValueError(
                    f"Candidate positions are not a complete permutation for {impression_id}"
                )
            if len(set(article_ids)) != len(article_ids):
                raise ValueError(f"Candidate article IDs are not unique for {impression_id}")
            rank_by_position = {position: rank + 1 for rank, position in enumerate(positions)}
            ranks = [rank_by_position[position] for position in range(len(positions))]
            raw_id = str(impression_id).rsplit(":", 1)[-1]
            handle.write(f"{raw_id} [{','.join(map(str, ranks))}]\n")
            count += 1
    return count


def write_mind_submission(predictions: pl.DataFrame, output: Path) -> None:
    write_competition_submission(predictions, output)


def _archive_submission(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as destination:
        destination.write(output, arcname=output.name)
    return archive


def submit_from_config(config: Path | Mapping[str, object]) -> tuple[Path, Path]:
    mapping = dict(config) if isinstance(config, Mapping) else load_mapping(config)
    dataset = str(mapping["name"])
    variant = str(mapping["variant"])
    system = str(mapping["system"])
    predictions_path = project_path(
        str(
            mapping.get(
                "predictions",
                Path("data")
                / "retrieval"
                / dataset
                / variant
                / system
                / "competition_test"
                / "impression_candidates.parquet",
            )
        )
    )
    predictions = pl.read_parquet(predictions_path)
    data_root = project_path(str(mapping.get("data", "data")))
    expected_path = (
        data_root / "processed" / dataset / variant / "competition_test" / "impressions.parquet"
    )
    expected_count = pq.ParquetFile(expected_path).metadata.num_rows
    output = project_path(
        str(
            mapping.get(
                "submission",
                Path("output") / "submissions" / f"{dataset}-{variant}-{system}.txt",
            )
        )
    )
    actual_count = write_competition_submission(predictions, output)
    if actual_count != expected_count:
        raise ValueError(
            f"Submission contains {actual_count} impressions; expected {expected_count}"
        )
    archive = _archive_submission(output)
    print(output)
    print(archive)
    return output, archive
