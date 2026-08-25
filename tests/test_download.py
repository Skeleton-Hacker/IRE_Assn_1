import io
from pathlib import Path
from urllib.request import Request

import pytest

from ire_assn1.data.configuration import parse_dataset_config
from ire_assn1.data.download import DownloadedArchive, _download_file, download_dataset


def test_download_file_uses_configured_token(monkeypatch, tmp_path: Path) -> None:
    requests: list[Request] = []

    def open_url(request: Request) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO(b"archive")

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    destination = tmp_path / "archive.zip"

    _download_file("https://huggingface.co/datasets/example/archive.zip", destination, "HF_TOKEN")

    assert destination.read_bytes() == b"archive"
    assert requests[0].get_header("Authorization") == "Bearer test-token"


def test_download_file_requires_configured_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    destination = tmp_path / "archive.zip"

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _download_file(
            "https://huggingface.co/datasets/example/archive.zip",
            destination,
            "HF_TOKEN",
        )

    assert not destination.exists()


def test_download_ebnerd_includes_submission_artifacts(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path, Path]] = []

    def download(url: str, archive: Path, extracted: Path, auth_env: str | None = None):
        assert auth_env is None
        calls.append((url, archive, extracted))
        return DownloadedArchive(url, archive, extracted)

    monkeypatch.setattr("ire_assn1.data.download._download_archive", download)
    config = parse_dataset_config(
        {
            "name": "ebnerd",
            "variant": "large",
            "language": "da",
            "archive_url": "https://example.test/large.zip",
            "archive": "large.zip",
            "articles_archive_url": "https://example.test/articles.zip",
            "articles_archive": "articles.zip",
            "test_archive_url": "https://example.test/test.zip",
            "test_archive": "test.zip",
            "embedding_archive_url": "https://example.test/bert.zip",
            "embedding_archive": "bert.zip",
            "embedding_path": "data/raw/ebnerd/large/embeddings/bert",
            "embedding_source": "bert",
            "availability": "published_at",
        }
    )

    result = download_dataset(config, tmp_path)

    assert len(result) == 4
    assert [call[2].relative_to(tmp_path).as_posix() for call in calls] == [
        "raw/ebnerd/large/extracted",
        "raw/ebnerd/large/articles",
        "raw/ebnerd/large/competition_test",
        "raw/ebnerd/large/embeddings/bert",
    ]
