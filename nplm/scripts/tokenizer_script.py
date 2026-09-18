# nplm/scripts/tokenizer_script.py
"""
Provides the command-line entry point for building and saving the configured tokenizer.
"""
import argparse
from pathlib import Path

from nplm.config import ConfigLoader
from nplm.orchestrators.tokenization_orchestrator import TokenizationOrchestrator
from nplm.tokenization.word_tokenizer import TokenizerConfig


def main():
    """main: loads the tokenizer configuration and executes the tokenizer-building workflow."""
    parser = argparse.ArgumentParser(description="Build the configured tokenizer.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_loader = ConfigLoader()
    config = config_loader.load(args.config)
    tokenizer_config_data = config_loader.require_section(config, "tokenization")

    tokenizer_config = TokenizerConfig(
        min_freq=tokenizer_config_data.get("min_freq", 2),
        max_vocab=tokenizer_config_data.get("max_vocab", 20000),
        lowercase=tokenizer_config_data.get("lowercase", False),
        tokenizer=tokenizer_config_data.get("tokenizer", "simple"),
        strip_punct=tokenizer_config_data.get("strip_punct", False),
        include_bos=tokenizer_config_data.get("include_bos", True),
        include_eos=tokenizer_config_data.get("include_eos", True),
    )

    orchestrator = TokenizationOrchestrator()

    tokenizer, artifact_path = orchestrator.run(
        tokenizer_type=tokenizer_config_data.get("type", "word"),
        jsonl_dir=Path(tokenizer_config_data["jsonl_dir"]),
        output_path=Path(tokenizer_config_data["output_path"]),
        config=tokenizer_config,
        text_field=tokenizer_config_data.get("text_field", "text"),
        progress=tokenizer_config_data.get("progress", True),
    )

    print(f"Tokenizer built with {len(tokenizer.id_to_token)} tokens.")
    print(f"Tokenizer saved to: {artifact_path}")


if __name__ == "__main__":
    main()