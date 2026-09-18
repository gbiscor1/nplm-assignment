# tests/nplm/tokenization/test_tokenizer_factory.py
"""
Tests the tokenizer factory and verifies that configured tokenizer implementations are built and loaded correctly.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nplm.tokenization.tokenizer_factory import TokenizerFactory
from nplm.tokenization.word_tokenizer import TokenizerConfig


@patch("nplm.tokenization.tokenizer_factory.WordTokenizer")
def test_load_word_tokenizer(mock_word_tokenizer):
    """Tests that the factory loads and returns a configured word tokenizer."""
    mock_tokenizer = MagicMock()
    mock_word_tokenizer.load.return_value = mock_tokenizer
    tokenizer_path = Path("artifacts/wikitext2_word_vocab.json")

    result = TokenizerFactory.load_tokenizer(
        tokenizer_type="word",
        tokenizer_path=tokenizer_path,
    )

    mock_word_tokenizer.load.assert_called_once_with(tokenizer_path)
    assert result is mock_tokenizer


@patch("nplm.tokenization.tokenizer_factory.WordTokenizer")
def test_load_tokenizer_unsupported_type(mock_word_tokenizer):
    """Tests that an unsupported tokenizer type raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        TokenizerFactory.load_tokenizer(
            tokenizer_type="invalid",
            tokenizer_path=Path("artifacts/tokenizer.json"),
        )

    mock_word_tokenizer.load.assert_not_called()


@patch("nplm.tokenization.tokenizer_factory.WordTokenizer")
def test_load_tokenizer_normalizes_type(mock_word_tokenizer):
    """Tests that tokenizer type capitalization and surrounding whitespace are normalized."""
    mock_tokenizer = MagicMock()
    mock_word_tokenizer.load.return_value = mock_tokenizer
    tokenizer_path = Path("artifacts/wikitext2_word_vocab.json")

    result = TokenizerFactory.load_tokenizer(
        tokenizer_type="  Word  ",
        tokenizer_path=tokenizer_path,
    )

    mock_word_tokenizer.load.assert_called_once_with(tokenizer_path)
    assert result is mock_tokenizer


@patch("nplm.tokenization.tokenizer_factory.WordTokenizer")
def test_build_word_tokenizer(mock_word_tokenizer):
    """Tests that the factory builds and returns a configured word tokenizer."""
    mock_tokenizer = MagicMock()
    mock_word_tokenizer.build_from_corpus.return_value = mock_tokenizer

    tokenizer_config = TokenizerConfig(min_freq=2, max_vocab=20000)
    jsonl_dir = Path("data/wikitext2_jsonl/train")

    result = TokenizerFactory.build_tokenizer(
        tokenizer_type="word",
        jsonl_dir=jsonl_dir,
        config=tokenizer_config,
        text_field="text",
        progress=False,
    )

    mock_word_tokenizer.build_from_corpus.assert_called_once_with(
        jsonl_dir=str(jsonl_dir),
        config=tokenizer_config,
        text_field="text",
        progress=False,
    )
    assert result is mock_tokenizer


@patch("nplm.tokenization.tokenizer_factory.WordTokenizer")
def test_build_tokenizer_unsupported_type(mock_word_tokenizer):
    """Tests that building an unsupported tokenizer type raises a ValueError."""
    tokenizer_config = TokenizerConfig()

    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        TokenizerFactory.build_tokenizer(
            tokenizer_type="invalid",
            jsonl_dir=Path("data/train"),
            config=tokenizer_config,
        )

    mock_word_tokenizer.build_from_corpus.assert_not_called()


@patch("nplm.tokenization.tokenizer_factory.WordTokenizer")
def test_build_tokenizer_normalizes_type(mock_word_tokenizer):
    """Tests that tokenizer type normalization is applied when building a tokenizer."""
    mock_tokenizer = MagicMock()
    mock_word_tokenizer.build_from_corpus.return_value = mock_tokenizer

    tokenizer_config = TokenizerConfig()
    jsonl_dir = Path("data/train")

    result = TokenizerFactory.build_tokenizer(
        tokenizer_type="  Word  ",
        jsonl_dir=jsonl_dir,
        config=tokenizer_config,
    )

    mock_word_tokenizer.build_from_corpus.assert_called_once_with(
        jsonl_dir=str(jsonl_dir),
        config=tokenizer_config,
        text_field="text",
        progress=True,
    )
    assert result is mock_tokenizer