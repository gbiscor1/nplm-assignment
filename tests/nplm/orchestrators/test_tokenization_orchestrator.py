# tests/nplm/orchestrators/test_tokenization_orchestrator.py
"""
Tests the tokenization orchestrator and verifies that the configured tokenizer is built and saved correctly.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nplm.orchestrators.tokenization_orchestrator import TokenizationOrchestrator
from nplm.tokenization.word_tokenizer import TokenizerConfig


@patch("nplm.orchestrators.tokenization_orchestrator.TokenizerFactory.build_tokenizer")
def test_run_tokenization(mock_build_tokenizer):
    """Tests that the orchestrator builds the configured tokenizer and saves its artifact."""
    mock_tokenizer = MagicMock()
    output_path = Path("artifacts/wikitext2_word_vocab.json")
    mock_tokenizer.save.return_value = output_path
    mock_build_tokenizer.return_value = mock_tokenizer

    tokenizer_config = TokenizerConfig(min_freq=2, max_vocab=20000)
    jsonl_dir = Path("data/wikitext2_jsonl/train")

    orchestrator = TokenizationOrchestrator()

    tokenizer, artifact_path = orchestrator.run(
        tokenizer_type="word",
        jsonl_dir=jsonl_dir,
        output_path=output_path,
        config=tokenizer_config,
        text_field="text",
        progress=False,
    )

    mock_build_tokenizer.assert_called_once_with(
        tokenizer_type="word",
        jsonl_dir=jsonl_dir,
        config=tokenizer_config,
        text_field="text",
        progress=False,
    )
    mock_tokenizer.save.assert_called_once_with(output_path)

    assert tokenizer is mock_tokenizer
    assert artifact_path == output_path


@patch("nplm.orchestrators.tokenization_orchestrator.TokenizerFactory.build_tokenizer")
def test_run_tokenization_factory_failure(mock_build_tokenizer):
    """Tests that tokenizer factory failures are propagated by the orchestrator."""
    mock_build_tokenizer.side_effect = ValueError("Unsupported tokenizer type.")

    orchestrator = TokenizationOrchestrator()

    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        orchestrator.run(
            tokenizer_type="invalid",
            jsonl_dir=Path("data/wikitext2_jsonl/train"),
            output_path=Path("artifacts/tokenizer.json"),
            config=TokenizerConfig(),
        )


@patch("nplm.orchestrators.tokenization_orchestrator.TokenizerFactory.build_tokenizer")
def test_run_tokenization_uses_default_optional_values(mock_build_tokenizer):
    """Tests that the orchestrator passes the expected default optional values to the factory."""
    mock_tokenizer = MagicMock()
    output_path = Path("artifacts/wikitext2_word_vocab.json")
    mock_tokenizer.save.return_value = output_path
    mock_build_tokenizer.return_value = mock_tokenizer

    tokenizer_config = TokenizerConfig()
    jsonl_dir = Path("data/wikitext2_jsonl/train")

    orchestrator = TokenizationOrchestrator()

    tokenizer, artifact_path = orchestrator.run(
        tokenizer_type="word",
        jsonl_dir=jsonl_dir,
        output_path=output_path,
        config=tokenizer_config,
    )

    mock_build_tokenizer.assert_called_once_with(
        tokenizer_type="word",
        jsonl_dir=jsonl_dir,
        config=tokenizer_config,
        text_field="text",
        progress=True,
    )
    mock_tokenizer.save.assert_called_once_with(output_path)

    assert tokenizer is mock_tokenizer
    assert artifact_path == output_path