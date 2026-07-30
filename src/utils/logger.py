"""
Logging configuration utility.
"""

import logging
import logging.config
import os

import yaml


def setup_logger(
    default_path: str = "config/logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
) -> logging.Logger:
    """Configures project logger using YAML config file."""
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, "rt", encoding="utf-8") as f:
            config = yaml.safe_load(f.read())

        # Ensure log directory exists
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                os.makedirs(os.path.dirname(handler["filename"]), exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    return logging.getLogger("bayesian_cox")
