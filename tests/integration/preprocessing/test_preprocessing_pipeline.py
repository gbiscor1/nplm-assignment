# tests/integration/preprocessing/test_preprocessing_pipeline.py
"""
Tests the complete preprocessing pipeline using temporary raw input and output directories.
"""
import json

import pytest

from nplm.orchestrators.preprocessing_orchestrator import PreprocessingOrchestrator
from nplm.preprocessing.preprocess import Preprocessor


@pytest.mark.integration
def test_preprocessing_pipeline(tmp_path):
    """Tests that raw JSONL data is processed into train, validation, and test JSONL shards."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    input_file = input_dir / "raw.jsonl"

    input_dir.mkdir(parents=True, exist_ok=True)

    raw_records = [
        {"text": "  Document One  "},
        {"text": "DOCUMENT TWO"},
        {"text": "Document Three"},
        {"text": "Document Four"},
        {"text": "Document Five"},
        {"text": "Document Six"},
        {"text": "Document Seven"},
        {"text": "Document Eight"},
        {"text": "Document Nine"},
        {"text": "Document Ten"},
    ]

    with input_file.open("w", encoding="utf-8") as file:
        for record in raw_records:
            file.write(json.dumps(record) + "\n")

    preprocessor = Preprocessor(lowercase=True, seed=42)
    orchestrator = PreprocessingOrchestrator(preprocessor)

    result = orchestrator.run(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        shard_size=2,
    )

    train_dir = output_dir / "train"
    validation_dir = output_dir / "validation"
    test_dir = output_dir / "test"

    assert train_dir.exists()
    assert validation_dir.exists()
    assert test_dir.exists()

    assert len(result["train"]) == 3
    assert len(result["validation"]) == 1
    assert len(result["test"]) == 1

    assert all(path.exists() for path in result["train"])
    assert all(path.exists() for path in result["validation"])
    assert all(path.exists() for path in result["test"])

    processed_records = []

    for split_paths in result.values():
        for shard_path in split_paths:
            with shard_path.open("r", encoding="utf-8") as file:
                for line in file:
                    processed_records.append(json.loads(line))

    assert len(processed_records) == 10
    assert all("text" in record for record in processed_records)
    assert all(record["text"] == record["text"].strip() for record in processed_records)
    assert all(record["text"] == record["text"].lower() for record in processed_records)