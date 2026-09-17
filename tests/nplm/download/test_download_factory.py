# tests/nplm/download/test_download_factory.py
"""
Tests the dataset downloader factory and verifies that downloader implementations are selected and configured correctly.
"""
from unittest.mock import MagicMock, patch

import pytest

from nplm.download.download_factory import create_downloader


@patch("nplm.download.download_factory.HuggingFaceDownloader")
def test_create_huggingface_downloader(mock_huggingface_downloader):
    """Tests that the factory creates and returns a configured Hugging Face downloader."""
    mock_downloader = MagicMock()
    mock_huggingface_downloader.return_value = mock_downloader

    result = create_downloader(
        backend="huggingface",
        dataset_name="wikitext",
        subset_name="wikitext-2-raw-v1",
        text_field="text",
    )

    mock_huggingface_downloader.assert_called_once_with(
        dataset_name="wikitext",
        subset_name="wikitext-2-raw-v1",
        text_field="text",
    )
    assert result is mock_downloader


@patch("nplm.download.download_factory.HuggingFaceDownloader")
def test_create_downloader_unsupported_backend(mock_huggingface_downloader):
    """Tests that an unsupported backend raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported download backend"):
        create_downloader(
            backend="invalid",
            dataset_name="wikitext",
        )

    mock_huggingface_downloader.assert_not_called()


@patch("nplm.download.download_factory.HuggingFaceDownloader")
def test_create_downloader_normalizes_backend(mock_huggingface_downloader):
    """Tests that backend capitalization and surrounding whitespace are normalized."""
    mock_downloader = MagicMock()
    mock_huggingface_downloader.return_value = mock_downloader

    result = create_downloader(
        backend="  HuggingFace  ",
        dataset_name="wikitext",
        subset_name=None,
        text_field="text",
    )

    mock_huggingface_downloader.assert_called_once_with(
        dataset_name="wikitext",
        subset_name=None,
        text_field="text",
    )
    assert result is mock_downloader