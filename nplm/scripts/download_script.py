# nplm/scripts/download_script.py
"""
Provides the command-line entry point for downloading a configured dataset.
"""
import argparse
from pathlib import Path

from nplm.config import ConfigLoader
from nplm.orchestrators.download_orchestrator import DownloadOrchestrator


def main():
    """main: loads the download configuration and executes the dataset download workflow."""
    parser = argparse.ArgumentParser(description="Download the configured dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_loader = ConfigLoader()
    config = config_loader.load(args.config)
    download_config = config_loader.require_section(config, "download")

    orchestrator = DownloadOrchestrator()

    output_dir = orchestrator.run(
        backend=download_config["backend"],
        dataset_name=download_config["dataset_name"],
        output_dir=Path(download_config["output_dir"]),
        subset_name=download_config.get("subset_name"),
        text_field=download_config.get("text_field", "text"),
    )

    print(f"Dataset downloaded to: {output_dir}")


if __name__ == "__main__":
    main()