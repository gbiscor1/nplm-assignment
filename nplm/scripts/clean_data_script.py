# nplm/scripts/clean_data_script.py
"""
Provides the command-line entry point for removing generated downloaded and preprocessed dataset files.
"""
import argparse
import shutil
from pathlib import Path

from nplm.config import ConfigLoader


def _remove_directory(path: Path) -> bool:
    """
    _remove_directory: removes a generated data directory if it exists.

    Params:
        path (Path): generated directory to remove.

    Returns:
        bool (bool): whether a directory was removed.
    """
    if not path.exists():
        return False

    if not path.is_dir():
        raise ValueError(f"Expected generated data path to be a directory: '{path}'.")

    shutil.rmtree(path)

    return True


def main():
    """main: loads configured generated data paths and removes downloaded and preprocessed dataset directories."""
    parser = argparse.ArgumentParser(description="Remove generated downloaded and preprocessed dataset files.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_loader = ConfigLoader()
    config = config_loader.load(args.config)

    download_config = config_loader.require_section(config, "download")
    preprocessing_config = config_loader.require_section(config, "preprocessing")

    download_dir = Path(download_config["output_dir"])
    preprocessing_dir = Path(preprocessing_config["output_dir"])

    removed_download = _remove_directory(download_dir)
    removed_preprocessing = _remove_directory(preprocessing_dir)

    if removed_download:
        print(f"Removed downloaded data: {download_dir}")
    else:
        print(f"Downloaded data directory does not exist: {download_dir}")

    if removed_preprocessing:
        print(f"Removed preprocessed data: {preprocessing_dir}")
    else:
        print(f"Preprocessed data directory does not exist: {preprocessing_dir}")


if __name__ == "__main__":
    main()