from __future__ import annotations

from pathlib import Path

import polars as pl

from ire_assn1.paths import project_path
from ire_assn1.settings import load_mapping


def write_mind_submission(predictions: pl.DataFrame, output: Path) -> None:
    required = {"impression_id", "position", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"MIND predictions are missing columns: {', '.join(sorted(missing))}")
    ranked = predictions.sort(
        ["impression_id", "score", "article_id"],
        descending=[False, True, False],
    )
    grouped = ranked.group_by("impression_id", maintain_order=True).agg(pl.col("position"))
    with output.open("w", encoding="utf-8") as handle:
        for impression_id, positions in grouped.iter_rows():
            ordered = list(positions)
            rank_by_position = {position: rank + 1 for rank, position in enumerate(ordered)}
            ranks = [rank_by_position[position] for position in sorted(ordered)]
            raw_id = str(impression_id).rsplit(":", 1)[-1]
            handle.write(f"{raw_id} [{','.join(map(str, ranks))}]\n")


def submit_from_config(config: Path) -> None:
    mapping = load_mapping(config)
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
                / "impression_candidates.parquet",
            )
        )
    )
    predictions = pl.read_parquet(predictions_path)
    if "source_split" in predictions.columns:
        predictions = predictions.filter(pl.col("source_split") == "test")
    output = project_path(
        str(
            mapping.get(
                "submission",
                Path("output") / "submissions" / f"{dataset}-{system}.txt",
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "mind":
        write_mind_submission(predictions, output)
    elif "submission_template" in mapping:
        template = pl.read_parquet(project_path(mapping["submission_template"]))
        joined = template.drop("score", strict=False).join(
            predictions.select("impression_id", "article_id", "score"),
            on=["impression_id", "article_id"],
            how="left",
        )
        joined.write_parquet(output)
    else:
        raise ValueError("EB-NeRD submission requires the current Codabench template")
    print(output)
