# tests/nplm/tokenization/test_word_tokenizer.py
"""
Tests the word tokenizer implementation with emphasis on the shared tokenizer interface wrappers and artifact persistence behavior.
"""
from pathlib import Path

from nplm.tokenization.word_tokenizer import BOS, EOS, UNK, TokenizerConfig, WordTokenizer


def _build_test_tokenizer() -> WordTokenizer:
    """Creates a small word tokenizer instance for unit testing."""
    config = TokenizerConfig(lowercase=True, include_bos=True, include_eos=True)

    token_to_id = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "!": 6,
    }

    id_to_token = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "hello",
        "world",
        "!",
    ]

    return WordTokenizer(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        config=config,
        freqs={"hello": 10, "world": 5, "!": 3},
    )


def test_encode_wrapper():
    """Tests that encode uses the existing text encoding behavior."""
    tokenizer = _build_test_tokenizer()

    result = tokenizer.encode("Hello world!")

    assert result == [4, 5, 6]


def test_encode_wrapper_unknown_token():
    """Tests that encode maps unknown words to the unknown token ID."""
    tokenizer = _build_test_tokenizer()

    result = tokenizer.encode("Hello unknownword")

    assert result == [4, tokenizer.unk_id]


def test_decode_wrapper():
    """Tests that decode converts token IDs into a space-separated text representation."""
    tokenizer = _build_test_tokenizer()

    result = tokenizer.decode([4, 5, 6])

    assert result == "hello world !"


def test_decode_wrapper_invalid_id():
    """Tests that decode converts invalid token IDs to the unknown token."""
    tokenizer = _build_test_tokenizer()

    result = tokenizer.decode([4, 999])

    assert result == f"hello {UNK}"


def test_save_wrapper(tmp_path):
    """Tests that save writes the tokenizer artifact and returns its path."""
    tokenizer = _build_test_tokenizer()
    output_path = tmp_path / "tokenizer.json"

    result = tokenizer.save(output_path)

    assert result == output_path
    assert output_path.exists()


def test_save_wrapper_creates_parent_directory(tmp_path):
    """Tests that save creates missing parent directories."""
    tokenizer = _build_test_tokenizer()
    output_path = tmp_path / "artifacts" / "tokenizer.json"

    result = tokenizer.save(output_path)

    assert result == output_path
    assert output_path.exists()


def test_load_saved_tokenizer(tmp_path):
    """Tests that a saved tokenizer can be loaded with the same vocabulary and configuration."""
    tokenizer = _build_test_tokenizer()
    output_path = tmp_path / "tokenizer.json"

    tokenizer.save(output_path)
    loaded = WordTokenizer.load(output_path)

    assert loaded.token_to_id == tokenizer.token_to_id
    assert loaded.id_to_token == tokenizer.id_to_token
    assert loaded.config == tokenizer.config
    assert loaded.freqs == tokenizer.freqs


def test_encode_text_with_bos_eos():
    """Tests that the existing text encoding method can include boundary tokens."""
    tokenizer = _build_test_tokenizer()

    result = tokenizer.encode_text("Hello world", with_bos_eos=True)

    assert result == [
        tokenizer.token_to_id[BOS],
        tokenizer.token_to_id["hello"],
        tokenizer.token_to_id["world"],
        tokenizer.token_to_id[EOS],
    ]


def test_tokenize_strip_punctuation():
    """Tests that punctuation is removed when strip_punct is enabled."""
    config = TokenizerConfig(lowercase=True, strip_punct=True)

    tokenizer = WordTokenizer(
        token_to_id={"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "hello": 4, "world": 5},
        id_to_token=["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"],
        config=config,
    )

    result = tokenizer.tokenize("Hello, world!")

    assert result == ["hello", "world"]