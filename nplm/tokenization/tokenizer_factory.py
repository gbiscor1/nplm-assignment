# nplm/tokenization/tokenizer_factory.py
"""
Creates and loads tokenizer implementations based on the configured tokenizer type.
"""
from pathlib import Path

from nplm.tokenization.tokenization_interface import Tokenizer
from nplm.tokenization.word_tokenizer import TokenizerConfig, WordTokenizer


class TokenizerFactory:
    """
    Class TokenizerFactory: creates and loads tokenizer implementations based on the configured tokenizer type.

    Internal parameters:
        None
    """

    @staticmethod
    def build_tokenizer(tokenizer_type: str, jsonl_dir: Path, config: TokenizerConfig, text_field: str = "text", progress: bool = True) -> Tokenizer:
        """
        build_tokenizer: builds the tokenizer implementation associated with the requested tokenizer type.

        Params:
            tokenizer_type (str): name of the configured tokenizer type.
            jsonl_dir (Path): directory containing the training JSONL shards.
            config (TokenizerConfig): configuration used to build the tokenizer.
            text_field (str): name of the dataset field containing document text.
            progress (bool): whether progress should be displayed while scanning the corpus.

        Returns:
            Tokenizer (Tokenizer): built tokenizer implementation.
        """
        tokenizer_type = tokenizer_type.strip().lower()

        if tokenizer_type == "word":
            return WordTokenizer.build_from_corpus(jsonl_dir=str(jsonl_dir), config=config, text_field=text_field, progress=progress)

        raise ValueError(f"Unsupported tokenizer type: '{tokenizer_type}'.")

    @staticmethod
    def load_tokenizer(tokenizer_type: str, tokenizer_path: Path) -> Tokenizer:
        """
        load_tokenizer: loads the tokenizer implementation associated with the requested tokenizer type.

        Params:
            tokenizer_type (str): name of the configured tokenizer type.
            tokenizer_path (Path): path to the saved tokenizer artifact.

        Returns:
            Tokenizer (Tokenizer): loaded tokenizer implementation.
        """
        tokenizer_type = tokenizer_type.strip().lower()

        if tokenizer_type == "word":
            return WordTokenizer.load(tokenizer_path)

        raise ValueError(f"Unsupported tokenizer type: '{tokenizer_type}'.")