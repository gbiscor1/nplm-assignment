# tests/integration/tokenization/test_tokenization_pipeline.py
"""
Tests the complete tokenization pipeline by building, saving, loading, and using a word tokenizer with temporary JSONL training data.
"""
import json

import pytest

from nplm.orchestrators.tokenization_orchestrator import TokenizationOrchestrator
from nplm.tokenization.tokenizer_factory import TokenizerFactory
from nplm.tokenization.word_tokenizer import TokenizerConfig


@pytest.mark.integration
def test_tokenization_pipeline(tmp_path):
    """Tests that the complete tokenization pipeline builds, saves, loads, and uses a word tokenizer."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    shard_path = train_dir / "shard_0.jsonl"
    records = [
        {"text": "hello world"},
        {"text": "hello tokenizer"},
        {"text": "world hello"},
    ]

    with shard_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    output_path = tmp_path / "artifacts" / "word_vocab.json"

    tokenizer_config = TokenizerConfig(
        min_freq=2,
        max_vocab=100,
        lowercase=False,
        tokenizer="simple",
        strip_punct=False,
        include_bos=True,
        include_eos=True,
    )

    orchestrator = TokenizationOrchestrator()

    tokenizer, artifact_path = orchestrator.run(
        tokenizer_type="word",
        jsonl_dir=train_dir,
        output_path=output_path,
        config=tokenizer_config,
        text_field="text",
        progress=False,
    )

    assert artifact_path == output_path
    assert output_path.exists()

    assert "hello" in tokenizer.token_to_id
    assert "world" in tokenizer.token_to_id
    assert "tokenizer" not in tokenizer.token_to_id

    loaded_tokenizer = TokenizerFactory.load_tokenizer(
        tokenizer_type="word",
        tokenizer_path=output_path,
    )

    assert loaded_tokenizer.token_to_id == tokenizer.token_to_id
    assert loaded_tokenizer.id_to_token == tokenizer.id_to_token

    encoded = loaded_tokenizer.encode("hello world")

    assert encoded == [
        loaded_tokenizer.token_to_id["hello"],
        loaded_tokenizer.token_to_id["world"],
    ]