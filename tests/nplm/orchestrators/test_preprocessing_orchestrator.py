# tests/nplm/orchestrators/test_preprocessing_orchestrator.py
"""
Tests the preprocessing orchestrator and verifies that preprocessing operations are executed in the expected order with the correct data.
"""
from pathlib import Path
from unittest.mock import MagicMock

from nplm.orchestrators.preprocessing_orchestrator import PreprocessingOrchestrator


def test_initialization():
    """Tests that the orchestrator stores the configured preprocessor."""
    mock_preprocessor = MagicMock()

    orchestrator = PreprocessingOrchestrator(mock_preprocessor)

    assert orchestrator.preprocessor is mock_preprocessor


def test_run_preprocessing():
    """Tests that the orchestrator executes the complete preprocessing workflow."""
    mock_preprocessor = MagicMock()

    raw_records = [{"text": "Raw document"}]
    documents = [{"text": "Processed document"}]
    splits = {
        "train": [{"text": "Train document"}],
        "validation": [{"text": "Validation document"}],
        "test": [{"text": "Test document"}],
    }

    train_paths = [Path("processed/train/shard_0.jsonl")]
    validation_paths = [Path("processed/validation/shard_0.jsonl")]
    test_paths = [Path("processed/test/shard_0.jsonl")]

    mock_preprocessor.read_raw_records.return_value = raw_records
    mock_preprocessor.prepare_documents.return_value = documents
    mock_preprocessor.split_dataset.return_value = splits
    mock_preprocessor.write_shards.side_effect = [train_paths, validation_paths, test_paths]

    orchestrator = PreprocessingOrchestrator(mock_preprocessor)

    input_file = Path("data/raw/wikitext2/raw.jsonl")
    output_dir = Path("data/wikitext2_jsonl")

    result = orchestrator.run(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        shard_size=10000,
    )

    mock_preprocessor.read_raw_records.assert_called_once_with(input_file)
    mock_preprocessor.prepare_documents.assert_called_once_with(raw_records)
    mock_preprocessor.split_dataset.assert_called_once_with(documents, 0.8, 0.1, 0.1)

    assert mock_preprocessor.write_shards.call_count == 3
    mock_preprocessor.write_shards.assert_any_call(splits["train"], output_dir / "train", 10000)
    mock_preprocessor.write_shards.assert_any_call(splits["validation"], output_dir / "validation", 10000)
    mock_preprocessor.write_shards.assert_any_call(splits["test"], output_dir / "test", 10000)

    assert result == {
        "train": train_paths,
        "validation": validation_paths,
        "test": test_paths,
    }


def test_run_preprocessing_read_failure():
    """Tests that raw dataset loading failures are propagated by the orchestrator."""
    mock_preprocessor = MagicMock()
    mock_preprocessor.read_raw_records.side_effect = ValueError("Invalid raw dataset.")

    orchestrator = PreprocessingOrchestrator(mock_preprocessor)

    try:
        orchestrator.run(
            input_file=Path("raw.jsonl"),
            output_dir=Path("processed"),
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            shard_size=100,
        )

        assert False, "Expected ValueError to be raised."

    except ValueError as error:
        assert str(error) == "Invalid raw dataset."

    mock_preprocessor.prepare_documents.assert_not_called()
    mock_preprocessor.split_dataset.assert_not_called()
    mock_preprocessor.write_shards.assert_not_called()


def test_run_preprocessing_empty_dataset():
    """Tests that an empty dataset still executes the split and shard workflow correctly."""
    mock_preprocessor = MagicMock()

    mock_preprocessor.read_raw_records.return_value = []
    mock_preprocessor.prepare_documents.return_value = []
    mock_preprocessor.split_dataset.return_value = {
        "train": [],
        "validation": [],
        "test": [],
    }
    mock_preprocessor.write_shards.side_effect = [[], [], []]

    orchestrator = PreprocessingOrchestrator(mock_preprocessor)
    output_dir = Path("processed")

    result = orchestrator.run(
        input_file=Path("raw.jsonl"),
        output_dir=output_dir,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        shard_size=100,
    )

    assert result == {
        "train": [],
        "validation": [],
        "test": [],
    }

    assert mock_preprocessor.write_shards.call_count == 3