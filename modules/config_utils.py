from pathlib import Path
from copy import deepcopy
import json
import random
import numpy as np
import torch


def deep_update(default: dict, override: dict) -> dict:
    """
    Recursively update default config using user config.
    Used to create final CONFIG from DEFAULT_CONFIG and USER_CONFIG.
    """
    ...


def load_json(path: str | Path) -> dict:
    """
    Load a JSON config file.
    """
    ...


def save_json(obj: dict, path: str | Path) -> None:
    """
    Save dictionary to JSON file with indentation.
    """
    ...


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for random, numpy, and torch.
    """
    ...


def ensure_dirs(*dirs: str | Path) -> None:
    """
    Create directories if they do not exist.
    """
    ...


def get_device():
    """
    Return torch.device('cuda') if available, otherwise torch.device('cpu').
    """
    ...


def print_config(config: dict) -> None:
    """
    Pretty print final configuration for notebook display.
    """
    ...