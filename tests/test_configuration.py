from ire_assn1.data.configuration import parse_dataset_config


def test_large_dataset_variants_are_supported() -> None:
    mind = parse_dataset_config(
        {
            "name": "mind",
            "variant": "large",
            "language": "en",
            "train_url": "https://example.test/train.zip",
            "validation_url": "https://example.test/dev.zip",
            "train_archive": "train.zip",
            "validation_archive": "dev.zip",
            "availability": "first_seen",
        }
    )
    ebnerd = parse_dataset_config(
        {
            "name": "ebnerd",
            "variant": "large",
            "language": "da",
            "archive_url": "https://example.test/ebnerd_large.zip",
            "archive": "ebnerd_large.zip",
            "availability": "published_at",
        }
    )

    assert mind.variant == "large"
    assert ebnerd.variant == "large"
