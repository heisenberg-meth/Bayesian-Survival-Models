"""
Configuration management system supporting YAML loading, dot-notation attribute access,
and hierarchical configuration merging.
"""

import os
from typing import Any, Dict, Union
import yaml


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot-notation access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")


def load_yaml(file_path: str) -> ConfigDict:
    """Loads a YAML configuration file into a ConfigDict object."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at '{file_path}'")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    return ConfigDict(data)


def merge_configs(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merges override configuration into base configuration."""
    merged = ConfigDict(base.copy())
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_project_config(config_dir: str = "config") -> ConfigDict:
    """Loads full hierarchical project configuration from config directory."""
    base_file = os.path.join(config_dir, "base.yaml")
    datasets_file = os.path.join(config_dir, "datasets.yaml")
    model_file = os.path.join(config_dir, "model.yaml")
    training_file = os.path.join(config_dir, "training.yaml")

    config = load_yaml(base_file) if os.path.exists(base_file) else ConfigDict()
    
    if os.path.exists(datasets_file):
        config.datasets = load_yaml(datasets_file)
    if os.path.exists(model_file):
        config.models = load_yaml(model_file)
    if os.path.exists(training_file):
        config.training = load_yaml(training_file)

    return config
