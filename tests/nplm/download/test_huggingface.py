# tests/nplm/download/test_huggingface.py
"""
Tests the Hugging Face downloader and verifies dataset loading, record extraction, and standardized output behavior.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nplm.download.huggingface import HuggingFaceDownloader


def test_huggingface_downloader_initialization():
    """Tests that the downloader stores the provided configuration values."""
    downloader = HuggingFaceDownloader(dataset_name="wikitext", subset_name="wikitext-2-raw-v1", text_field="content")

    assert downloader.dataset_name == "wikitext"
    assert downloader.subset_name == "wikitext-2-raw-v1"
    assert downloader.text_field == "content"


def test_huggingface_downloader_default_initialization():
    """Tests that optional downloader configuration uses the expected defaults."""
    downloader = HuggingFaceDownloader(dataset_name="wikitext")

    assert downloader.dataset_name == "wikitext"
    assert downloader.subset_name is None
    assert downloader.text_field == "text"


@patch("nplm.download.huggingface.write_jsonl")
@patch("nplm.download.huggingface.load_dataset")
def test_download(mock_load_dataset, mock_write_jsonl):
    """Tests that download loads the configured dataset and writes the standardized raw file."""
    dataset = {"train": [{"text": "First record."}]}
    mock_load_dataset.return_value = dataset

    downloader = HuggingFaceDownloader(dataset_name="wikitext", subset_name="wikitext-2-raw-v1", text_field="text")
    output_dir = Path("data/raw/wikitext2")

    result = downloader.download(output_dir)

    mock_load_dataset.assert_called_once_with("wikitext", "wikitext-2-raw-v1")
    mock_write_jsonl.assert_called_once()

    written_path, records = mock_write_jsonl.call_args.args

    assert written_path == output_dir / "raw.jsonl"
    assert list(records) == [{"text": "First record."}]
    assert result == output_dir


@patch("nplm.download.huggingface.write_jsonl")
@patch("nplm.download.huggingface.load_dataset")
def test_download_load_failure(mock_load_dataset, mock_write_jsonl):
    """Tests that dataset loading failures are propagated and no output is written."""
    mock_load_dataset.side_effect = RuntimeError("Dataset could not be loaded.")

    downloader = HuggingFaceDownloader(dataset_name="wikitext")

    with pytest.raises(RuntimeError, match="Dataset could not be loaded"):
        downloader.download(Path("data/raw/wikitext2"))

    mock_write_jsonl.assert_not_called()


@patch("nplm.download.huggingface.write_jsonl")
@patch("nplm.download.huggingface.load_dataset")
def test_download_empty_dataset(mock_load_dataset, mock_write_jsonl):
    """Tests that an empty dataset can still be passed to the standardized writer."""
    mock_load_dataset.return_value = {}

    downloader = HuggingFaceDownloader(dataset_name="wikitext")
    output_dir = Path("data/raw/wikitext2")

    result = downloader.download(output_dir)

    mock_write_jsonl.assert_called_once()

    written_path, records = mock_write_jsonl.call_args.args

    assert written_path == output_dir / "raw.jsonl"
    assert list(records) == []
    assert result == output_dir


def test_extract_records():
    """Tests that records from multiple dataset splits are converted to the standardized text format."""
    downloader = HuggingFaceDownloader(dataset_name="wikitext")
    dataset = {
        "train": [{"text": "Train one."}, {"text": "Train two."}],
        "validation": [{"text": "Validation one."}],
    }

    records = list(downloader._extract_records(dataset))

    assert records == [
        {"text": "Train one."},
        {"text": "Train two."},
        {"text": "Validation one."},
    ]


def test_extract_records_missing_text_field():
    """Tests that a record missing the configured text field raises a ValueError."""
    downloader = HuggingFaceDownloader(dataset_name="wikitext")
    dataset = {"train": [{"content": "Missing text field."}]}

    with pytest.raises(ValueError, match="Text field 'text' was not found"):
        list(downloader._extract_records(dataset))


def test_extract_records_invalid_text_type():
    """Tests that a non-string text field raises a TypeError."""
    downloader = HuggingFaceDownloader(dataset_name="wikitext")
    dataset = {"train": [{"text": 123}]}

    with pytest.raises(TypeError, match="Expected field 'text' to contain a string"):
        list(downloader._extract_records(dataset))


def test_extract_records_empty_dataset():
    """Tests that an empty dataset produces no records."""
    downloader = HuggingFaceDownloader(dataset_name="wikitext")

    records = list(downloader._extract_records({}))

    assert records == []