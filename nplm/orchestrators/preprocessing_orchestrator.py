# nplm/orchestrators/preprocessing_orchestrator.py
"""
Coordinates the preprocessing workflow by applying the configured preprocessing operations and writing the resulting dataset splits as JSONL shards.
"""
from pathlib import Path

from nplm.preprocessing.preprocess import Preprocessor


class PreprocessingOrchestrator:
    """
    Class PreprocessingOrchestrator: coordinates the complete dataset preprocessing workflow.

    Internal parameters:
        preprocessor (Preprocessor): preprocessing implementation used to prepare and shard the dataset.
    """

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def run(self, input_file: Path, output_dir: Path, train_ratio: float, validation_ratio: float, test_ratio: float, shard_size: int) -> dict[str, list[Path]]:
        """
        run: preprocesses the raw dataset and writes train, validation, and test JSONL shards.

        Params:
            input_file (Path): path to the standardized raw JSONL dataset file.
            output_dir (Path): directory where the processed dataset splits should be stored.
            train_ratio (float): proportion of documents assigned to the training split.
            validation_ratio (float): proportion of documents assigned to the validation split.
            test_ratio (float): proportion of documents assigned to the test split.
            shard_size (int): maximum number of documents written to each shard.

        Returns:
            dict[str, list[Path]] (dict[str, list[Path]]): shard paths created for each dataset split.
        """
        records = self.preprocessor.read_raw_records(input_file)
        documents = self.preprocessor.prepare_documents(records)
        splits = self.preprocessor.split_dataset(documents, train_ratio, validation_ratio, test_ratio)

        return {
            "train": self.preprocessor.write_shards(splits["train"], output_dir / "train", shard_size),
            "validation": self.preprocessor.write_shards(splits["validation"], output_dir / "validation", shard_size),
            "test": self.preprocessor.write_shards(splits["test"], output_dir / "test", shard_size),
        }