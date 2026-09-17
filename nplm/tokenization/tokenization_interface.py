# nplm\tokenization\tokenization_interface.py
"""
Defines the common interface for tokenizer implementations used by the NPLM application.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):
    """
    Class Tokenizer: abstract class for tokenizer implementations.

    Internal parameters:
        None
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        encode: converts text into token IDs.

        Params:
            text (str): text to tokenize and encode.

        Returns:
            list[int] (list[int]): encoded token IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        decode: converts token IDs back into text.

        Params:
            token_ids (list[int]): token IDs to decode.

        Returns:
            str (str): decoded text.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, output_path: Path) -> Path:
        """
        save: saves the tokenizer artifact to disk.

        Params:
            output_path (Path): destination path for the tokenizer artifact.

        Returns:
            Path (Path): path to the saved tokenizer artifact.
        """
        raise NotImplementedError