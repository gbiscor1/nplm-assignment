# nplm/preprocessing/preprocess.py
"""
Provides reusable operations for reading raw dataset records, cleaning text, creating reproducible dataset splits, and writing processed JSONL shards.
"""
import json
import math
import random
from pathlib import Path

from nplm.utils.dataset_formatter import write_jsonl


class Preprocessor:
    """
    Class Preprocessor: provides reusable operations for preparing raw text datasets for model training and evaluation.

    Internal parameters:
        lowercase (bool): whether document text should be converted to lowercase.
        seed (int): random seed used for reproducible dataset splitting.
    """

    def __init__(self, lowercase: bool = False, seed: int = 1337):
        self.lowercase = lowercase
        self.seed = seed

    def read_raw_records(self, input_file: Path) -> list[dict]:
        """
        read_raw_records: reads standardized raw JSONL records from a dataset file.

        Params:
            input_file (Path): path to the standardized raw JSONL file.

        Returns:
            list[dict] (list[dict]): records loaded from the raw dataset file.
        """
        # Safeguard againist empty directory
        if not input_file.exists():
            raise FileNotFoundError(f"Raw dataset file was not found: '{input_file}'.")

        # Record container
        records = []

        with input_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON found on line {line_number} of '{input_file}'.") from error

                if "text" not in record:
                    raise ValueError(f"Record on line {line_number} of '{input_file}' does not contain a 'text' field.")

                if not isinstance(record["text"], str):
                    raise TypeError(f"Expected 'text' field on line {line_number} of '{input_file}' to contain a string.")

                records.append(record)

        return records

    def clean_text(self, text: str) -> str:
        """
        clean_text: applies the configured text cleaning operations to a document.

        Params:
            text (str): document text to clean.

        Returns:
            str (str): cleaned document text.
        """
        text = text.strip()

        if self.lowercase:
            text = text.lower()

        return text

    def prepare_documents(self, records: list[dict]) -> list[dict]:
        """
        prepare_documents: converts raw records into cleaned non-empty document records.

        Params:
            records (list[dict]): raw dataset records containing text fields.

        Returns:
            list[dict] (list[dict]): cleaned non-empty document records.
        """
        documents = []

        for record in records:
            text = self.clean_text(record["text"])

            if text:
                documents.append({"text": text})

        return documents

    def split_dataset(self, documents: list[dict], train_ratio: float, validation_ratio: float, test_ratio: float) -> dict[str, list[dict]]:
        """
        split_dataset: creates reproducible train, validation, and test splits from the processed documents.

        Params:
            documents (list[dict]): processed document records to split.
            train_ratio (float): proportion of documents assigned to the training split.
            validation_ratio (float): proportion of documents assigned to the validation split.
            test_ratio (float): proportion of documents assigned to the test split.

        Returns:
            dict[str, list[dict]] (dict[str, list[dict]]): train, validation, and test document splits.
        """
        total_ratio = train_ratio + validation_ratio + test_ratio

        if not math.isclose(total_ratio, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1.0.")

        if train_ratio < 0 or validation_ratio < 0 or test_ratio < 0:
            raise ValueError("Dataset split ratios cannot be negative.")

        shuffled_documents = documents.copy()
        random.Random(self.seed).shuffle(shuffled_documents)

        total_documents = len(shuffled_documents)
        train_end = int(total_documents * train_ratio)
        validation_end = train_end + int(total_documents * validation_ratio)

        return {
            "train": shuffled_documents[:train_end],
            "validation": shuffled_documents[train_end:validation_end],
            "test": shuffled_documents[validation_end:],
        }

    def write_shards(self, documents: list[dict], output_dir: Path, shard_size: int) -> list[Path]:
        """
        write_shards: writes processed documents into JSONL shard files with a configurable maximum number of documents per shard.

        Params:
            documents (list[dict]): processed documents to write.
            output_dir (Path): directory where the JSONL shards should be stored.
            shard_size (int): maximum number of documents written to each shard.

        Returns:
            list[Path] (list[Path]): paths to the JSONL shard files that were created.
        """
        if shard_size <= 0:
            raise ValueError("Shard size must be greater than zero.")

        shard_paths = []

        for shard_index, start_index in enumerate(range(0, len(documents), shard_size)):
            shard_documents = documents[start_index:start_index + shard_size]
            shard_path = output_dir / f"shard_{shard_index}.jsonl"

            write_jsonl(shard_path, shard_documents)
            shard_paths.append(shard_path)

        return shard_paths