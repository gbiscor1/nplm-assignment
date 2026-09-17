# nplm/scripts/preprocess_script.py
"""
Provides the command-line entry point for preprocessing a configured dataset.
"""
import argparse
from pathlib import Path

from nplm.config import ConfigLoader
from nplm.orchestrators.preprocessing_orchestrator import PreprocessingOrchestrator
from nplm.preprocessing.preprocess import Preprocessor


def main():
    """main: loads the preprocessing configuration and executes the preprocessing workflow."""
    parser = argparse.ArgumentParser(description="Preprocess the configured dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_loader = ConfigLoader()
    config = config_loader.load(args.config)
    preprocessing_config = config_loader.require_section(config, "preprocessing")

    preprocessor = Preprocessor(
        lowercase=preprocessing_config.get("lowercase", False),
        seed=preprocessing_config.get("seed", 1337),
    )

    orchestrator = PreprocessingOrchestrator(preprocessor)

    shard_paths = orchestrator.run(
        input_file=Path(preprocessing_config["input_file"]),
        output_dir=Path(preprocessing_config["output_dir"]),
        train_ratio=preprocessing_config["train_ratio"],
        validation_ratio=preprocessing_config["validation_ratio"],
        test_ratio=preprocessing_config["test_ratio"],
        shard_size=preprocessing_config["shard_size"],
    )

    print(f"Preprocessing complete. Train shards: {len(shard_paths['train'])}, Validation shards: {len(shard_paths['validation'])}, Test shards: {len(shard_paths['test'])}")


if __name__ == "__main__":
    main()