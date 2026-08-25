from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from ire_assn1.experiments.ebnerd_submission import write_ebnerd_semantic_submission


def test_ebnerd_semantic_submission_preserves_rows_and_ranks(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "ebnerd" / "large"
    test = raw / "competition_test" / "ebnerd_testset" / "test"
    embeddings = raw / "embeddings" / "bert" / "google_bert"
    test.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "user_id": [10, 11],
                "article_id_fixed": [[1], [2]],
            }
        ),
        test / "history.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "impression_id": [99, 7, 50],
                "user_id": [10, 11, 12],
                "article_ids_inview": [[2, 3], [1, 3, 2], [3, 1]],
            }
        ),
        test / "behaviors.parquet",
        row_group_size=2,
    )
    embedding_path = embeddings / "bert_base_multilingual_cased.parquet"
    pq.write_table(
        pa.table(
            {
                "article_id": [1, 2, 3],
                "google-bert/bert-base-multilingual-cased": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.2],
                ],
            }
        ),
        embedding_path,
    )
    output = tmp_path / "submission" / "predictions.txt"
    mapping = {
        "name": "ebnerd",
        "variant": "large",
        "data": str(tmp_path),
        "embedding_path": str(embeddings),
        "submission_history_length": 20,
        "submission_batch_size": 2,
        "submission_device": "cpu",
        "submission": str(output),
    }

    text_path, archive_path = write_ebnerd_semantic_submission(mapping)

    assert text_path == output
    assert output.read_text(encoding="utf-8") == ("99 [2,1]\n7 [3,2,1]\n50 [2,1]\n")
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["predictions.txt"]
        assert archive.read("predictions.txt").decode() == output.read_text(encoding="utf-8")

    assert write_ebnerd_semantic_submission(mapping) == (text_path, archive_path)
