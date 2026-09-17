# nplm/download/huggingface.py
"""
Downloads datasets from Hugging Face, extracts the configured text field, and saves the resulting records using the standardized raw dataset format used by the project.
"""
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from nplm.download.download_interface import Downloader
from nplm.utils.dataset_formatter import write_jsonl


class HuggingFaceDownloader(Downloader):
    """
    Class HuggingFaceDownloader: downloads and standardizes datasets from Hugging Face.

    Internal parameters:
        dataset_name (str): Hugging Face dataset identifier.
        subset_name (str | None): optional Hugging Face dataset configuration or subset name.
        text_field (str): name of the dataset field containing the text to extract.
    """

    def __init__(self, dataset_name: str, subset_name: str | None = None, text_field: str = "text"):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.text_field = text_field

    def download(self, output_dir: Path) -> Path:
        """
        download: downloads the configured Hugging Face dataset and saves it in the standardized raw JSONL format.

        Params:
            output_dir (Path): directory where the raw dataset should be stored.

        Returns:
            Path (Path): path to the downloaded dataset directory.
        """
        dataset = load_dataset(self.dataset_name, self.subset_name)

        output_path = output_dir / "raw.jsonl"
        write_jsonl(output_path, self._extract_records(dataset))

        return output_dir

    def _extract_records(self, dataset) -> Iterable[dict]:
        """
        _extract_records: extracts text records from all Hugging Face dataset splits into the standardized record format.

        Params:
            dataset: Hugging Face dataset object containing one or more dataset splits.

        Returns:
            Iterable[dict] (Iterable[dict]): standardized dataset records containing a text field.
        """
        for split_name, split_data in dataset.items():
            for record in split_data:
                if self.text_field not in record:
                    raise ValueError(f"Text field '{self.text_field}' was not found in dataset split '{split_name}'.")

                text = record[self.text_field]

                if not isinstance(text, str):
                    raise TypeError(f"Expected field '{self.text_field}' to contain a string, but received {type(text).__name__}.")

                yield {"text": text}