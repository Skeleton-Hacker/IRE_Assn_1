import io
from pathlib import Path
from urllib.request import Request

import pytest

from ire_assn1.data.download import _download_file


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
