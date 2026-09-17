# tests/integration/download/test_huggingface_download_pipeline.py
"""
Tests the complete Hugging Face download pipeline using the real external dataset source and a temporary output directory.
"""
import json

import pytest

from nplm.orchestrators.download_orchestrator import DownloadOrchestrator


@pytest.mark.integration
def test_huggingface_download_pipeline(tmp_path):
    """Tests that the complete Hugging Face download pipeline creates a standardized raw JSONL file."""
    output_dir = tmp_path / "wikitext2"

    orchestrator = DownloadOrchestrator()

    result = orchestrator.run(
        backend="huggingface",
        dataset_name="wikitext",
        output_dir=output_dir,
        subset_name="wikitext-2-raw-v1",
        text_field="text",
    )

    output_file = output_dir / "raw.jsonl"

    assert result == output_dir
    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as file:
        first_line = file.readline()

    first_record = json.loads(first_line)

    assert "text" in first_record
    assert isinstance(first_record["text"], str)