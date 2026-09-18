# tests/nplm/orchestrators/test_download_orchestrator.py
"""
Tests the download orchestrator and verifies that the configured downloader is created and executed correctly.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nplm.orchestrators.download_orchestrator import DownloadOrchestrator


@patch("nplm.orchestrators.download_orchestrator.DownloadFactory.create_downloader")
def test_run_download(mock_create_downloader):
    """Tests that the orchestrator creates the configured downloader and executes the download."""
    mock_downloader = MagicMock()
    mock_downloader.download.return_value = Path("data/raw/wikitext2")
    mock_create_downloader.return_value = mock_downloader

    orchestrator = DownloadOrchestrator()
    output_dir = Path("data/raw/wikitext2")

    result = orchestrator.run(
        backend="huggingface",
        dataset_name="wikitext",
        output_dir=output_dir,
        subset_name="wikitext-2-raw-v1",
        text_field="text",
    )

    mock_create_downloader.assert_called_once_with(
        backend="huggingface",
        dataset_name="wikitext",
        subset_name="wikitext-2-raw-v1",
        text_field="text",
    )
    mock_downloader.download.assert_called_once_with(output_dir)
    assert result == output_dir


@patch("nplm.orchestrators.download_orchestrator.DownloadFactory.create_downloader")
def test_run_download_factory_failure(mock_create_downloader):
    """Tests that downloader factory failures are propagated by the orchestrator."""
    mock_create_downloader.side_effect = ValueError("Unsupported download backend.")

    orchestrator = DownloadOrchestrator()

    with pytest.raises(ValueError, match="Unsupported download backend"):
        orchestrator.run(
            backend="invalid",
            dataset_name="wikitext",
            output_dir=Path("data/raw/wikitext2"),
        )


@patch("nplm.orchestrators.download_orchestrator.DownloadFactory.create_downloader")
def test_run_download_uses_default_optional_values(mock_create_downloader):
    """Tests that the orchestrator passes the expected default optional values to the factory."""
    mock_downloader = MagicMock()
    mock_downloader.download.return_value = Path("data/raw/wikitext2")
    mock_create_downloader.return_value = mock_downloader

    orchestrator = DownloadOrchestrator()
    output_dir = Path("data/raw/wikitext2")

    result = orchestrator.run(
        backend="huggingface",
        dataset_name="wikitext",
        output_dir=output_dir,
    )

    mock_create_downloader.assert_called_once_with(
        backend="huggingface",
        dataset_name="wikitext",
        subset_name=None,
        text_field="text",
    )
    mock_downloader.download.assert_called_once_with(output_dir)
    assert result == output_dir