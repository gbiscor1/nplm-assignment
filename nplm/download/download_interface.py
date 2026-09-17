# nplm/download/interface.py
"""
Defines the common interface for dataset downloaders.

This module provides the abstract downloader contract used by the download orchestration layer so different download backends can be interchanged without changing the rest of the pipeline.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class Downloader(ABC):
    """
    Class Downloader: abstract class for dataset download implementations.

    Internal parameters:
        None
    """

    @abstractmethod
    def download(self, output_dir: Path) -> Path:
        """
        download: downloads a dataset to the specified output directory.

        Params:
            output_dir (Path): directory where the raw dataset should be stored.

        Returns:
            Path (Path): path to the downloaded dataset directory.
        """
        raise NotImplementedError