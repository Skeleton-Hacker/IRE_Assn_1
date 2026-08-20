from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from ire_assn1.data.configuration import (
    DatasetConfig,
    EbnerdDatasetConfig,
    MindDatasetConfig,
    load_data_config,
)
from ire_assn1.data.ebnerd import (
    prepare_ebnerd,
    prepare_ebnerd_competition_streaming,
    prepare_ebnerd_streaming,
)
from ire_assn1.data.mind import (
    prepare_mind,
    prepare_mind_competition_streaming,
    prepare_mind_streaming,
)
from ire_assn1.data.models import PreparationResult
from ire_assn1.data.store import write_feature_store


def prepare_dataset(config: DatasetConfig, data_root: Path) -> PreparationResult:
    raw_root = data_root / "raw" / config.name / config.variant
    if isinstance(config, MindDatasetConfig):
        if config.variant == "large":
            return prepare_mind_streaming(
                train_root=raw_root / "train",
                official_validation_root=raw_root / "official_validation",
                variant=config.variant,
                validation_days=config.validation_days,
                data_root=data_root,
            )
        result = prepare_mind(
            train_root=raw_root / "train",
            official_validation_root=raw_root / "official_validation",
            variant=config.variant,
            validation_days=config.validation_days,
        )
    elif isinstance(config, EbnerdDatasetConfig):
        if config.variant == "large":
            return prepare_ebnerd_streaming(
                extracted_root=raw_root / "extracted",
                variant=config.variant,
                validation_days=config.validation_days,
                data_root=data_root,
            )
        result = prepare_ebnerd(
            extracted_root=raw_root / "extracted",
            variant=config.variant,
            validation_days=config.validation_days,
        )
    else:
        raise TypeError(f"Unsupported dataset configuration: {type(config)}")
    return write_feature_store(result, data_root)


def prepare_from_config(config_path: str | Path) -> tuple[PreparationResult, ...]:
    data_root, datasets = load_data_config(config_path)
    return tuple(
        prepare_dataset(dataset, data_root)
        for dataset in tqdm(datasets, desc="prepare datasets", unit="dataset")
    )


def prepare_competition_from_config(config_path: str | Path) -> tuple[PreparationResult, ...]:
    data_root, datasets = load_data_config(config_path)
    results: list[PreparationResult] = []
    for dataset in tqdm(datasets, desc="prepare competition datasets", unit="dataset"):
        raw_root = data_root / "raw" / dataset.name / dataset.variant
        if isinstance(dataset, MindDatasetConfig):
            if dataset.variant != "large" or dataset.test_archive is None:
                continue
            results.append(
                prepare_mind_competition_streaming(
                    train_root=raw_root / "train",
                    official_validation_root=raw_root / "official_validation",
                    competition_test_root=raw_root / "competition_test",
                    variant=dataset.variant,
                    validation_days=dataset.validation_days,
                    data_root=data_root,
                )
            )
        elif isinstance(dataset, EbnerdDatasetConfig):
            if dataset.variant != "large" or dataset.test_archive is None:
                continue
            results.append(
                prepare_ebnerd_competition_streaming(
                    offline_extracted_root=raw_root / "extracted",
                    competition_root=raw_root / "competition_test",
                    variant=dataset.variant,
                    validation_days=dataset.validation_days,
                    data_root=data_root,
                )
            )
        else:
            raise TypeError(f"Unsupported dataset configuration: {type(dataset)}")
    return tuple(results)
