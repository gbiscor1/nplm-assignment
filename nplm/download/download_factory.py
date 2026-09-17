# nplm/download/download_factory.py
"""
Creates dataset downloader implementations based on the configured download backend.
"""
from nplm.download.huggingface import HuggingFaceDownloader
from nplm.download.download_interface import Downloader


def create_downloader(backend: str, dataset_name: str, subset_name: str | None = None, text_field: str = "text") -> Downloader:
    """
    create_downloader: creates the downloader implementation associated with the requested backend.

    Params:
        backend (str): name of the configured download backend.
        dataset_name (str): dataset identifier used by the selected downloader.
        subset_name (str | None): optional dataset subset or configuration name.
        text_field (str): name of the dataset field containing the text to extract.

    Returns:
        Downloader (Downloader): configured downloader implementation.
    """
    backend = backend.strip().lower()

    if backend == "huggingface":
        return HuggingFaceDownloader(dataset_name=dataset_name, subset_name=subset_name, text_field=text_field)

    raise ValueError(f"Unsupported download backend: '{backend}'.")