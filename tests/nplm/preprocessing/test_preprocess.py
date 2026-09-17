# tests/nplm/preprocessing/test_preprocess.py
"""
Tests the preprocessing operations for raw record loading, text cleaning, dataset splitting, document preparation, and JSONL sharding.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nplm.preprocessing.preprocess import Preprocessor


def test_initialization():
    """Tests that the preprocessor stores the configured options."""
    preprocessor = Preprocessor(lowercase=True, seed=42)

    assert preprocessor.lowercase is True
    assert preprocessor.seed == 42


def test_read_raw_records(tmp_path):
    """Tests that valid raw JSONL records are loaded correctly."""
    input_file = tmp_path / "raw.jsonl"
    records = [
        {"text": "First document."},
        {"text": "Second document."},
    ]

    with input_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    preprocessor = Preprocessor()
    result = preprocessor.read_raw_records(input_file)

    assert result == records


def test_read_raw_records_invalid_json(tmp_path):
    """Tests that invalid JSON raises a ValueError."""
    input_file = tmp_path / "raw.jsonl"
    input_file.write_text('{"text": "Valid"}\n{invalid json}\n', encoding="utf-8")

    preprocessor = Preprocessor()

    with pytest.raises(ValueError, match="Invalid JSON found on line 2"):
        preprocessor.read_raw_records(input_file)


def test_read_raw_records_missing_text_field(tmp_path):
    """Tests that records without a text field raise a ValueError."""
    input_file = tmp_path / "raw.jsonl"
    input_file.write_text('{"content": "Missing text field"}\n', encoding="utf-8")

    preprocessor = Preprocessor()

    with pytest.raises(ValueError, match="does not contain a 'text' field"):
        preprocessor.read_raw_records(input_file)


def test_read_raw_records_invalid_text_type(tmp_path):
    """Tests that non-string text fields raise a TypeError."""
    input_file = tmp_path / "raw.jsonl"
    input_file.write_text('{"text": 123}\n', encoding="utf-8")

    preprocessor = Preprocessor()

    with pytest.raises(TypeError, match="to contain a string"):
        preprocessor.read_raw_records(input_file)


def test_read_raw_records_skips_empty_lines(tmp_path):
    """Tests that empty lines in the raw JSONL file are ignored."""
    input_file = tmp_path / "raw.jsonl"
    input_file.write_text('\n{"text": "Document"}\n\n', encoding="utf-8")

    preprocessor = Preprocessor()
    result = preprocessor.read_raw_records(input_file)

    assert result == [{"text": "Document"}]

def test_read_raw_records_missing_file(tmp_path):
    """Tests that a missing raw dataset file raises a FileNotFoundError."""
    input_file = tmp_path / "missing.jsonl"
    preprocessor = Preprocessor()

    with pytest.raises(FileNotFoundError, match="Raw dataset file was not found"):
        preprocessor.read_raw_records(input_file)


def test_clean_text():
    """Tests that text is stripped and lowercased when configured."""
    preprocessor = Preprocessor(lowercase=True)

    result = preprocessor.clean_text("   Hello, World!   ")

    assert result == "hello, world!"


def test_clean_text_without_lowercase():
    """Tests that cleaning preserves capitalization when lowercasing is disabled."""
    preprocessor = Preprocessor(lowercase=False)

    result = preprocessor.clean_text("   Hello, World!   ")

    assert result == "Hello, World!"


def test_clean_text_empty():
    """Tests that whitespace-only text becomes an empty string."""
    preprocessor = Preprocessor()

    result = preprocessor.clean_text("     ")

    assert result == ""


def test_prepare_documents():
    """Tests that records are cleaned and empty documents are removed."""
    preprocessor = Preprocessor(lowercase=True)

    records = [
        {"text": "  First Document  "},
        {"text": "   "},
        {"text": "SECOND DOCUMENT"},
    ]

    result = preprocessor.prepare_documents(records)

    assert result == [
        {"text": "first document"},
        {"text": "second document"},
    ]


def test_prepare_documents_empty_input():
    """Tests that an empty record collection produces no documents."""
    preprocessor = Preprocessor()

    result = preprocessor.prepare_documents([])

    assert result == []


def test_split_dataset():
    """Tests that documents are split according to the configured proportions."""
    preprocessor = Preprocessor(seed=42)
    documents = [{"text": str(index)} for index in range(10)]

    result = preprocessor.split_dataset(documents, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)

    assert len(result["train"]) == 6
    assert len(result["validation"]) == 2
    assert len(result["test"]) == 2


def test_split_dataset_reproducible():
    """Tests that the same seed produces the same dataset split."""
    documents = [{"text": str(index)} for index in range(10)]

    first = Preprocessor(seed=42).split_dataset(documents, 0.6, 0.2, 0.2)
    second = Preprocessor(seed=42).split_dataset(documents, 0.6, 0.2, 0.2)

    assert first == second


def test_split_dataset_invalid_ratio_sum():
    """Tests that split ratios not summing to one raise a ValueError."""
    preprocessor = Preprocessor()

    with pytest.raises(ValueError, match="must sum to 1.0"):
        preprocessor.split_dataset([], train_ratio=0.7, validation_ratio=0.2, test_ratio=0.2)


def test_split_dataset_negative_ratio():
    """Tests that negative split ratios raise a ValueError."""
    preprocessor = Preprocessor()

    with pytest.raises(ValueError, match="cannot be negative"):
        preprocessor.split_dataset([], train_ratio=0.8, validation_ratio=0.3, test_ratio=-0.1)


def test_split_dataset_empty_input():
    """Tests that an empty dataset produces empty splits."""
    preprocessor = Preprocessor()

    result = preprocessor.split_dataset([], train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)

    assert result == {
        "train": [],
        "validation": [],
        "test": [],
    }


@patch("nplm.preprocessing.preprocess.write_jsonl")
def test_write_shards(mock_write_jsonl):
    """Tests that documents are divided into the expected JSONL shards."""
    preprocessor = Preprocessor()
    documents = [{"text": str(index)} for index in range(5)]
    output_dir = Path("output")

    result = preprocessor.write_shards(documents, output_dir, shard_size=2)

    assert result == [
        output_dir / "shard_0.jsonl",
        output_dir / "shard_1.jsonl",
        output_dir / "shard_2.jsonl",
    ]

    assert mock_write_jsonl.call_count == 3

    mock_write_jsonl.assert_any_call(output_dir / "shard_0.jsonl", documents[0:2])
    mock_write_jsonl.assert_any_call(output_dir / "shard_1.jsonl", documents[2:4])
    mock_write_jsonl.assert_any_call(output_dir / "shard_2.jsonl", documents[4:5])


@patch("nplm.preprocessing.preprocess.write_jsonl")
def test_write_shards_invalid_size(mock_write_jsonl):
    """Tests that non-positive shard sizes raise a ValueError."""
    preprocessor = Preprocessor()

    with pytest.raises(ValueError, match="greater than zero"):
        preprocessor.write_shards([], Path("output"), shard_size=0)

    mock_write_jsonl.assert_not_called()


@patch("nplm.preprocessing.preprocess.write_jsonl")
def test_write_shards_empty_documents(mock_write_jsonl):
    """Tests that an empty document collection creates no shard files."""
    preprocessor = Preprocessor()

    result = preprocessor.write_shards([], Path("output"), shard_size=10)

    assert result == []
    mock_write_jsonl.assert_not_called()
