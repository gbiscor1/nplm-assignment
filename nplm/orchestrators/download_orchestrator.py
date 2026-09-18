# nplm/orchestrators/download_orchestrator.py
"""
Coordinates the dataset download workflow by creating the configured downloader and executing the download process.
"""
from pathlib import Path

from nplm.download.download_factory import DownloadFactory


class DownloadOrchestrator:
    """
    Class DownloadOrchestrator: coordinates the dataset download workflow.

    Internal parameters:
        None
    """

    def run(self, backend: str, dataset_name: str, output_dir: Path, subset_name: str | None = None, text_field: str = "text") -> Path:
        """
        run: creates the configured downloader and executes the dataset download workflow.

        Params:
            backend (str): name of the configured download backend.
            dataset_name (str): dataset identifier used by the selected downloader.
            output_dir (Path): directory where the raw dataset should be stored.
            subset_name (str | None): optional dataset subset or configuration name.
            text_field (str): name of the dataset field containing the text to extract.

        Returns:
            Path (Path): path to the downloaded dataset directory.
        """
        downloader = DownloadFactory.create_downloader(
            backend=backend,
            dataset_name=dataset_name,
            subset_name=subset_name,
            text_field=text_field,
        )

        return downloader.download(output_dir)