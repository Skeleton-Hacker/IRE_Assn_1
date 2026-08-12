from datetime import datetime

import pyarrow as pa
import pytest

from ire_assn1.data.contracts import (
    ARTICLES_SCHEMA,
    CANDIDATES_SCHEMA,
    HISTORIES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    RETRIEVAL_SCHEMA,
    table_from_rows,
    validate_table,
)
from ire_assn1.data.models import DatasetIdentity


def test_normalized_schema_fields_are_stable() -> None:
    assert ARTICLES_SCHEMA.names == [
        "article_id",
        "title",
        "abstract",
        "body",
        "category",
        "subcategory",
        "entities",
        "published_at",
        "available_at",
        "source_split",
    ]
    assert HISTORIES_SCHEMA.names == [
        "user_id",
        "article_ids",
        "timestamps",
        "source_split",
    ]
    assert IMPRESSIONS_SCHEMA.names == [
        "impression_id",
        "user_id",
        "session_id",
        "timestamp",
        "candidate_ids",
        "clicked_ids",
        "labels",
        "source_split",
    ]
    assert CANDIDATES_SCHEMA.names == [
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
        "label",
        "position",
        "source_split",
    ]
    assert RETRIEVAL_SCHEMA.names[-3:] == ["system", "score", "rank"]
    assert ARTICLES_SCHEMA.field("available_at").nullable is False
    assert HISTORIES_SCHEMA.field("timestamps").type.value_field.nullable is True


def test_dataset_identity_namespaces_every_identifier() -> None:
    identity = DatasetIdentity(name="mind", variant="small")
    assert identity.article_id("N1") == "mind:small:article:N1"
    assert identity.user_id("U1") == "mind:small:user:U1"
    assert identity.impression_id("1") == "mind:small:impression:1"
    assert identity.session_id("7") == "mind:small:session:7"


def test_table_builder_enforces_required_fields() -> None:
    timestamp = datetime(2019, 11, 13, 9)
    table = table_from_rows(
        [
            {
                "article_id": "mind:small:article:N1",
                "title": "Title",
                "abstract": "Abstract",
                "body": None,
                "category": None,
                "subcategory": None,
                "entities": [],
                "published_at": None,
                "available_at": timestamp,
                "source_split": "train",
            }
        ],
        ARTICLES_SCHEMA,
    )
    assert validate_table(table, ARTICLES_SCHEMA).num_rows == 1
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_table(pa.table({"article_id": ["N1"]}), ARTICLES_SCHEMA)
    with pytest.raises(ValueError, match="available_at"):
        table_from_rows(
            [
                {
                    "article_id": "mind:small:article:N1",
                    "title": "Title",
                    "abstract": "Abstract",
                    "body": None,
                    "category": None,
                    "subcategory": None,
                    "entities": [],
                    "published_at": None,
                    "available_at": None,
                    "source_split": "train",
                }
            ],
            ARTICLES_SCHEMA,
        )
