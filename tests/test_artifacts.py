from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ire_assn1.retrieval.artifacts import load_provided_embeddings


def test_load_provided_bert_embeddings_maps_article_ids(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ebnerd" / "large"
    root.mkdir(parents=True)
    table = pa.table(
        {
            "article_id": pa.array([2, 1], type=pa.int64()),
            "bert_embedding": pa.array([[0.0, 2.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            "roberta_embedding": pa.array([[2.0, 0.0], [0.0, 1.0]], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, root / "artifacts.parquet")

    result = load_provided_embeddings(
        tmp_path,
        "ebnerd",
        "large",
        {
            "ebnerd:large:article:1": object(),
            "ebnerd:large:article:2": object(),
        },
        {"embedding_source": "bert"},
    )

    assert result.article_ids == (
        "ebnerd:large:article:1",
        "ebnerd:large:article:2",
    )
    assert result.vectors.tolist() == [[1.0, 0.0], [0.0, 2.0]]
