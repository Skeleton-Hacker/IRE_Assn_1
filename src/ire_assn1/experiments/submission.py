from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import pyarrow.parquet as pq

from ire_assn1.paths import project_path
from ire_assn1.settings import load_mapping


def _write_submission_rows(rows: Iterable[dict[str, Any]], output: Path) -> int:
    required = {"impression_id", "article_id", "position", "score"}
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    current_id: str | None = None
    current_rows: list[dict[str, Any]] = []

    def write_group(impression_id: str, group: list[dict[str, Any]]) -> None:
        positions = [int(row["position"]) for row in group]
        article_ids = [str(row["article_id"]) for row in group]
        complete = set(positions) == set(range(len(positions)))
        if len(set(positions)) != len(positions) or not complete:
            raise ValueError(
                f"Candidate positions are not a complete permutation for {impression_id}"
            )
        if len(set(article_ids)) != len(article_ids):
            raise ValueError(f"Candidate article IDs are not unique for {impression_id}")
        ranked = sorted(
            group,
            key=lambda row: (-float(row["score"]), str(row["article_id"]), int(row["position"])),
        )
        rank_by_position = {int(row["position"]): rank + 1 for rank, row in enumerate(ranked)}
        raw_id = impression_id.rsplit(":", 1)[-1]
        ranks = [rank_by_position[position] for position in range(len(positions))]
        handle.write(f"{raw_id} [{','.join(map(str, ranks))}]\n")

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            missing = required.difference(row)
            if missing:
                missing_columns = ", ".join(sorted(missing))
                raise ValueError(f"Competition predictions are missing columns: {missing_columns}")
            impression_id = str(row["impression_id"])
            if current_id is None:
                current_id = impression_id
            if impression_id != current_id:
                write_group(current_id, current_rows)
                count += 1
                current_id = impression_id
                current_rows = []
            current_rows.append(row)
        if current_id is not None:
            write_group(current_id, current_rows)
            count += 1
    return count


def write_competition_submission(predictions: pl.DataFrame, output: Path) -> int:
    return _write_submission_rows(predictions.to_dicts(), output)


def write_competition_submission_parquet(predictions: Path, output: Path) -> int:
    if not predictions.is_file():
        raise FileNotFoundError(predictions)

    def rows() -> Iterable[dict[str, Any]]:
        parquet = pq.ParquetFile(predictions)
        for batch in parquet.iter_batches(batch_size=8192):
            yield from cast(list[dict[str, Any]], batch.to_pylist())

    return _write_submission_rows(rows(), output)


def write_mind_submission(predictions: pl.DataFrame, output: Path) -> None:
    write_competition_submission(predictions, output)


def _archive_submission(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as destination:
        destination.write(output, arcname=output.name)
    return archive


def _default_submission_path(dataset: str, variant: str, system: str) -> Path:
    filename = "mind_prediction.txt" if dataset == "mind" else "predictions.txt"
    return Path("output") / "submissions" / dataset / variant / system / filename


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
    data_root = project_path(str(mapping.get("data", "data")))
    expected_path = (
        data_root / "processed" / dataset / variant / "competition_test" / "impressions.parquet"
    )
    expected_count = pq.ParquetFile(expected_path).metadata.num_rows
    output = project_path(
        str(
            mapping.get(
                "submission",
                _default_submission_path(dataset, variant, system),
            )
        )
    )
    actual_count = write_competition_submission_parquet(predictions_path, output)
    if actual_count != expected_count:
        raise ValueError(
            f"Submission contains {actual_count} impressions; expected {expected_count}"
        )
    archive = _archive_submission(output)
    print(output)
    print(archive)
    return output, archive
