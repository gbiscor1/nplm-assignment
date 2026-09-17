# nplm/config.py
"""
Loads and validates YAML configuration files used across the NPLM application.
"""
from pathlib import Path

import yaml


class ConfigLoader:
    """
    Class ConfigLoader: loads application configuration values from YAML files.

    Internal parameters:
        None
    """

    def load(self, config_path: Path) -> dict:
        """
        load: loads a YAML configuration file and returns its configuration values.

        Params:
            config_path (Path): path to the YAML configuration file.

        Returns:
            dict (dict): configuration values loaded from the YAML file.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file was not found: '{config_path}'.")

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Configuration file is empty: '{config_path}'.")

        if not isinstance(config, dict):
            raise TypeError(f"Configuration file must contain a YAML mapping: '{config_path}'.")

        return config

    def require_section(self, config: dict, section: str) -> dict:
        """
        require_section: retrieves a required configuration section and validates that it is a mapping.

        Params:
            config (dict): loaded application configuration.
            section (str): name of the required configuration section.

        Returns:
            dict (dict): configuration values contained in the requested section.
        """
        if section not in config:
            raise ValueError(f"Required configuration section was not found: '{section}'.")

        section_config = config[section]

        if not isinstance(section_config, dict):
            raise TypeError(f"Configuration section '{section}' must contain a YAML mapping.")

        return section_config