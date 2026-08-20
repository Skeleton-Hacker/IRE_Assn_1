from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ire_assn1.data.ebnerd import prepare_ebnerd, prepare_ebnerd_streaming
from ire_assn1.data.mind import prepare_mind, prepare_mind_streaming
from ire_assn1.data.pipeline import prepare_from_config
from ire_assn1.data.splits import official_temporal_split

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"


def test_official_temporal_split_uses_latest_complete_days() -> None:
    split = official_temporal_split(
        [
            datetime(2023, 6, 1, 22, tzinfo=UTC),
            datetime(2023, 6, 2, 9, tzinfo=UTC),
            datetime(2023, 6, 3, 8, tzinfo=UTC),
        ],
        validation_days=1,
    )
    assert split.validation_start == datetime(2023, 6, 3)
    assert split.assign_training(datetime(2023, 6, 2, 23, 59)) == "train"
    assert split.assign_training(datetime(2023, 6, 3)) == "validation"
    with pytest.raises(ValueError, match="earlier day"):
        official_temporal_split([datetime(2023, 6, 1)], validation_days=1)


def test_mind_ingestion_builds_official_splits_and_first_seen_availability() -> None:
    result = prepare_mind(
        FIXTURES / "mind" / "train",
        FIXTURES / "mind" / "official_validation",
    )
    assert result.stats.validation_start == datetime(2019, 11, 14)
    assert result.stats.article_duplicates == 1
    assert result.stats.article_conflicts == 0
    assert result.tables is not None
    impressions = result.tables.impressions.to_pylist()
    assert [row["source_split"] for row in impressions] == [
        "train",
        "validation",
        "validation",
        "test",
    ]
    articles = {row["article_id"]: row for row in result.tables.articles.to_pylist()}
    assert articles["mind:small:article:N1"]["available_at"] == datetime(2019, 11, 13, 9)
    assert articles["mind:small:article:N4"]["available_at"] == datetime(2019, 11, 15, 12)
    assert articles["mind:small:article:N1"]["entities"] == ["Q1"]
    candidates = result.tables.candidates.to_pylist()
    first = [row for row in candidates if row["impression_id"].endswith(":1")]
    assert [(row["position"], row["label"]) for row in first] == [(0, 1), (1, 0)]
    histories = result.tables.histories.to_pylist()
    assert {row["source_split"] for row in histories} == {"train", "validation", "test"}
    assert all(timestamp is None for row in histories for timestamp in row["timestamps"])


def test_large_mind_ingestion_streams_feature_store_rows(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "mind" / "large"
    shutil.copytree(FIXTURES / "mind" / "train", raw_root / "train")
    shutil.copytree(FIXTURES / "mind" / "official_validation", raw_root / "official_validation")

    result = prepare_mind_streaming(
        raw_root / "train",
        raw_root / "official_validation",
        variant="large",
        validation_days=1,
        data_root=tmp_path / "data",
    )

    assert result.tables is None
    assert result.row_counts == {
        "articles": 4,
        "candidates": 8,
        "histories": 4,
        "impressions": 4,
    }
    assert result.output_dir is not None
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == result.row_counts
    assert pq.read_table(result.output_dir / "impressions.parquet").num_rows == 4


def _write_ebnerd_fixture(root: Path) -> None:
    dataset = root / "ebnerd_demo"
    train = dataset / "train"
    validation = dataset / "validation"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "article_id": pa.array([1, 2, 3, 4], type=pa.int32()),
                "title": ["One", "Two", "Three", "Four"],
                "subtitle": ["First", "Second", "Third", "Fourth"],
                "body": ["Body one", "Body two", None, "Body four"],
                "category_str": ["news", "sport", "news", "culture"],
                "subcategory": [[10], [20, 21], [], [30]],
                "ner_clusters": [["A"], ["B"], [], ["C"]],
                "published_time": pa.array(
                    [
                        datetime(2023, 5, 1),
                        datetime(2023, 5, 2),
                        datetime(2023, 5, 3),
                        datetime(2023, 5, 4),
                    ],
                    type=pa.timestamp("us"),
                ),
            }
        ),
        dataset / "articles.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "impression_id": pa.array([10, 11], type=pa.uint32()),
                "user_id": pa.array([1, 2], type=pa.uint32()),
                "session_id": pa.array([100, 101], type=pa.uint32()),
                "impression_time": pa.array(
                    [datetime(2023, 6, 1, 9), datetime(2023, 6, 2, 9)],
                    type=pa.timestamp("us"),
                ),
                "article_ids_inview": pa.array([[1, 2], [2, 3]], type=pa.list_(pa.int32())),
                "article_ids_clicked": pa.array([[2], [3]], type=pa.list_(pa.int32())),
            }
        ),
        train / "behaviors.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([1, 2], type=pa.uint32()),
                "article_id_fixed": pa.array([[1], []], type=pa.list_(pa.int32())),
                "impression_time_fixed": pa.array(
                    [[datetime(2023, 5, 31, 9)], []],
                    type=pa.list_(pa.timestamp("us")),
                ),
            }
        ),
        train / "history.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "impression_id": pa.array([12], type=pa.uint32()),
                "user_id": pa.array([1], type=pa.uint32()),
                "session_id": pa.array([102], type=pa.uint32()),
                "impression_time": pa.array([datetime(2023, 6, 3, 9)], type=pa.timestamp("us")),
                "article_ids_inview": pa.array([[3, 4]], type=pa.list_(pa.int32())),
                "article_ids_clicked": pa.array([[4]], type=pa.list_(pa.int32())),
            }
        ),
        validation / "behaviors.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([1], type=pa.uint32()),
                "article_id_fixed": pa.array([[1, 2]], type=pa.list_(pa.int32())),
                "impression_time_fixed": pa.array(
                    [[datetime(2023, 5, 30), datetime(2023, 5, 31)]],
                    type=pa.list_(pa.timestamp("us")),
                ),
            }
        ),
        validation / "history.parquet",
    )


def test_ebnerd_ingestion_maps_official_parquet_columns(tmp_path: Path) -> None:
    _write_ebnerd_fixture(tmp_path)
    result = prepare_ebnerd(tmp_path, variant="demo")
    assert result.stats.validation_start == datetime(2023, 6, 2)
    assert result.tables is not None
    impressions = result.tables.impressions.to_pylist()
    assert [row["source_split"] for row in impressions] == [
        "train",
        "validation",
        "test",
    ]
    assert impressions[0]["session_id"] == "ebnerd:demo:session:100"
    assert impressions[0]["labels"] == [0, 1]
    articles = result.tables.articles.to_pylist()
    assert articles[0]["available_at"] == articles[0]["published_at"]
    assert articles[1]["subcategory"] == "20|21"
    histories = result.tables.histories.to_pylist()
    assert {(row["user_id"], row["source_split"]) for row in histories} == {
        ("ebnerd:demo:user:1", "train"),
        ("ebnerd:demo:user:2", "validation"),
        ("ebnerd:demo:user:1", "test"),
    }


def test_large_ebnerd_ingestion_streams_feature_store_rows(tmp_path: Path) -> None:
    _write_ebnerd_fixture(tmp_path)

    result = prepare_ebnerd_streaming(
        tmp_path,
        variant="large",
        validation_days=1,
        data_root=tmp_path / "data",
    )

    assert result.tables is None
    assert result.row_counts == {
        "articles": 4,
        "candidates": 6,
        "histories": 3,
        "impressions": 3,
    }
    assert result.output_dir is not None
    assert pq.read_table(result.output_dir / "articles.parquet").num_rows == 4


def test_config_driven_preparation_writes_feature_store(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw" / "mind" / "small"
    shutil.copytree(FIXTURES / "mind" / "train", raw_root / "train")
    shutil.copytree(FIXTURES / "mind" / "official_validation", raw_root / "official_validation")
    config_path = tmp_path / "mind.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: mind",
                "variant: small",
                "language: en",
                "train_url: https://example.test/train.zip",
                "validation_url: https://example.test/validation.zip",
                "train_archive: train.zip",
                "validation_archive: validation.zip",
                "availability: first_seen",
                "validation_days: 1",
                f"paths:\n  data: {data_root}",
            ]
        ),
        encoding="utf-8",
    )
    result = prepare_from_config(config_path)[0]
    assert result.output_dir == data_root / "processed" / "mind" / "small"
    assert result.output_dir is not None
    assert (result.output_dir / "articles.parquet").is_file()
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == {
        "articles": 4,
        "candidates": 8,
        "histories": 4,
        "impressions": 4,
    }
