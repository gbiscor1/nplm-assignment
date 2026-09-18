# nplm/orchestrators/tokenization_orchestrator.py
"""
Coordinates the tokenizer-building workflow by creating the configured tokenizer and saving the resulting tokenizer artifact.
"""
from pathlib import Path

from nplm.tokenization.tokenization_interface import Tokenizer
from nplm.tokenization.tokenizer_factory import TokenizerFactory
from nplm.tokenization.word_tokenizer import TokenizerConfig


class TokenizationOrchestrator:
    """
    Class TokenizationOrchestrator: coordinates tokenizer construction and artifact persistence.

    Internal parameters:
        None
    """

    def run(
        self,
        tokenizer_type: str,
        jsonl_dir: Path,
        output_path: Path,
        config: TokenizerConfig,
        text_field: str = "text",
        progress: bool = True,
    ) -> tuple[Tokenizer, Path]:
        """
        run: builds the configured tokenizer from training data and saves the resulting tokenizer artifact.

        Params:
            tokenizer_type (str): name of the tokenizer implementation to build.
            jsonl_dir (Path): directory containing the training JSONL shards.
            output_path (Path): destination path for the tokenizer artifact.
            config (TokenizerConfig): configuration used to build the tokenizer.
            text_field (str): name of the dataset field containing document text.
            progress (bool): whether progress should be displayed while scanning the corpus.

        Returns:
            tuple[Tokenizer, Path] (tuple[Tokenizer, Path]): built tokenizer and path to the saved tokenizer artifact.
        """
        tokenizer = TokenizerFactory.build_tokenizer(
            tokenizer_type=tokenizer_type,
            jsonl_dir=jsonl_dir,
            config=config,
            text_field=text_field,
            progress=progress,
        )

        artifact_path = tokenizer.save(output_path)

        return tokenizer, artifact_path