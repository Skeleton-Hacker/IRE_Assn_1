from __future__ import annotations

import json
import os
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

from tqdm.auto import tqdm

from ire_assn1.data.configuration import (
    DatasetConfig,
    EbnerdDatasetConfig,
    MindDatasetConfig,
    load_data_config,
)


@dataclass(frozen=True, slots=True)
class DownloadedArchive:
    url: str
    archive: Path
    extracted: Path


def _download_file(url: str, destination: Path, auth_env: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    token = os.environ.get(auth_env) if auth_env else None
    if auth_env and not token:
        raise RuntimeError(f"Missing {auth_env}; set it to a valid read token before downloading")
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    headers = {"User-Agent": "ire-assn1/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        try:
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                headers = getattr(response, "headers", None)
                content_length = headers.get("Content-Length") if headers is not None else None
                total = int(content_length) if content_length else None
                progress = tqdm(
                    total=total,
                    desc=f"download {destination.name}",
                    unit="B",
                    unit_scale=True,
                )
                try:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))
                finally:
                    progress.close()
        except HTTPError as error:
            if auth_env and error.code in {401, 403}:
                raise RuntimeError(
                    f"Dataset download returned HTTP {error.code} for {url}; "
                    f"check that {auth_env} contains a valid read token"
                ) from error
            raise RuntimeError(f"Dataset download returned HTTP {error.code}: {url}") from error
        except URLError as error:
            raise RuntimeError(f"Dataset download failed for {url}: {error.reason}") from error
        if temporary.stat().st_size == 0:
            raise ValueError(f"Downloaded archive is empty: {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_member(destination: Path, member: zipfile.ZipInfo) -> None:
    target = (destination / member.filename).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"Unsafe archive path: {member.filename}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"Archive symlinks are not supported: {member.filename}")


def _extract_archive(archive: Path, destination: Path, url: str) -> None:
    marker = destination / ".complete.json"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            _validate_member(destination, member)
        with tqdm(members, desc=f"extract {archive.name}", unit="file") as progress:
            for member in progress:
                source.extract(member, destination)
    payload = json.dumps({"archive": archive.name, "url": url}, sort_keys=True)
    marker.write_text(f"{payload}\n", encoding="utf-8")


def _download_archive(
    url: str,
    archive: Path,
    extracted: Path,
    auth_env: str | None = None,
) -> DownloadedArchive:
    _download_file(url, archive, auth_env=auth_env)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Invalid ZIP archive: {archive}")
    _extract_archive(archive, extracted, url)
    return DownloadedArchive(url=url, archive=archive, extracted=extracted)


def download_dataset(config: DatasetConfig, data_root: Path) -> tuple[DownloadedArchive, ...]:
    root = data_root / "raw" / config.name / config.variant
    archives = root / "archives"
    if isinstance(config, MindDatasetConfig):
        downloaded = [
            _download_archive(
                config.train_url,
                archives / config.train_archive,
                root / "train",
                auth_env=config.auth_env,
            ),
            _download_archive(
                config.validation_url,
                archives / config.validation_archive,
                root / "official_validation",
                auth_env=config.auth_env,
            ),
        ]
        if config.test_url and config.test_archive:
            downloaded.append(
                _download_archive(
                    config.test_url,
                    archives / config.test_archive,
                    root / "competition_test",
                    auth_env=config.auth_env,
                )
            )
        return tuple(downloaded)
    if isinstance(config, EbnerdDatasetConfig):
        downloaded = [
            _download_archive(
                config.archive_url,
                archives / config.archive,
                root / "extracted",
            ),
        ]
        if config.articles_archive_url and config.articles_archive:
            downloaded.append(
                _download_archive(
                    config.articles_archive_url,
                    archives / config.articles_archive,
                    root / "articles",
                )
            )
        if config.test_archive_url and config.test_archive:
            downloaded.append(
                _download_archive(
                    config.test_archive_url,
                    archives / config.test_archive,
                    root / "competition_test",
                )
            )
        if config.embedding_archive_url and config.embedding_archive:
            downloaded.append(
                _download_archive(
                    config.embedding_archive_url,
                    archives / config.embedding_archive,
                    root / "embeddings" / config.embedding_source,
                )
            )
        return tuple(downloaded)
    raise TypeError(f"Unsupported dataset configuration: {type(config)}")


def download_from_config(config_path: str | Path) -> tuple[DownloadedArchive, ...]:
    data_root, datasets = load_data_config(config_path)
    downloaded: list[DownloadedArchive] = []
    for dataset in datasets:
        downloaded.extend(download_dataset(dataset, data_root))
    return tuple(downloaded)


def download_ebnerd_submission_assets(
    config_path: str | Path,
) -> tuple[DownloadedArchive, ...]:
    data_root, datasets = load_data_config(config_path)
    matches = tuple(
        dataset
        for dataset in datasets
        if isinstance(dataset, EbnerdDatasetConfig) and dataset.variant == "large"
    )
    if len(matches) != 1:
        raise ValueError("Expected one EB-NeRD large dataset configuration")
    config = matches[0]
    if not config.test_archive_url or not config.test_archive:
        raise ValueError("EB-NeRD submission requires the official test archive")
    if not config.embedding_archive_url or not config.embedding_archive:
        raise ValueError("EB-NeRD semantic submission requires an embedding archive")
    root = data_root / "raw" / config.name / config.variant
    archives = root / "archives"
    return (
        _download_archive(
            config.test_archive_url,
            archives / config.test_archive,
            root / "competition_test",
        ),
        _download_archive(
            config.embedding_archive_url,
            archives / config.embedding_archive,
            root / "embeddings" / config.embedding_source,
        ),
    )
