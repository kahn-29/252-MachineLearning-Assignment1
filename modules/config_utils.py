# modules/config_utils.py

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def deep_update(default: dict, override: dict | None) -> dict:
    """
    Recursively update a default configuration dictionary with user-defined values.

    Parameters
    ----------
    default : dict
        The default configuration.
    override : dict | None
        The user configuration. Values in this dictionary override default values.

    Returns
    -------
    dict
        A new merged configuration dictionary.

    Example
    -------
    DEFAULT_CONFIG = {
        "pipeline": {
            "backbone": "efficientnet_b0",
            "classifier": "logistic_regression"
        }
    }

    USER_CONFIG = {
        "pipeline": {
            "classifier": "svm_linear"
        }
    }

    CONFIG = deep_update(DEFAULT_CONFIG, USER_CONFIG)
    """

    result = deepcopy(default)

    if override is None:
        return result

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_json(path: str | Path) -> dict:
    """
    Load a JSON file as a dictionary.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file.

    Returns
    -------
    dict
        Loaded JSON content.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str | Path) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    obj : dict
        Object to save.
    path : str | Path
        Output JSON path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.
    deterministic : bool
        If True, configures PyTorch for more deterministic behavior.
        This may reduce training speed.
    """

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def get_device():
    """
    Return the best available PyTorch device.

    Returns
    -------
    torch.device | str
        CUDA device if available, otherwise CPU.
        If PyTorch is not installed, returns "cpu".
    """

    if torch is None:
        return "cpu"

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs: str | Path) -> None:
    """
    Create directories if they do not already exist.

    Parameters
    ----------
    *dirs : str | Path
        One or more directory paths.
    """

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def create_project_dirs(base_dir: str | Path) -> dict[str, Path]:
    """
    Create and return standard project directories.

    Parameters
    ----------
    base_dir : str | Path
        Base working directory.

    Returns
    -------
    dict[str, Path]
        Dictionary of standard project directories.
    """

    base_dir = Path(base_dir)

    dirs = {
        "base": base_dir,
        "data": base_dir / "data",
        "features": base_dir / "features",
        "reports": base_dir / "reports",
        "figures": base_dir / "reports" / "figures",
        "results": base_dir / "results",
        "models": base_dir / "models",
        "logs": base_dir / "logs",
    }

    ensure_dirs(*dirs.values())

    return dirs


def print_config(config: dict, title: str = "Final Configuration") -> None:
    """
    Pretty print a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    title : str
        Display title.
    """

    print("=" * 80)
    print(title)
    print("=" * 80)
    pprint(config, sort_dicts=False)
    print("=" * 80)


def flatten_config(
    config: dict,
    parent_key: str = "",
    separator: str = "."
) -> dict:
    """
    Flatten a nested configuration dictionary.

    Parameters
    ----------
    config : dict
        Nested configuration dictionary.
    parent_key : str
        Internal prefix used during recursion.
    separator : str
        Separator between nested keys.

    Returns
    -------
    dict
        Flattened configuration.

    Example
    -------
    {"pipeline": {"backbone": "efficientnet_b0"}}
    becomes
    {"pipeline.backbone": "efficientnet_b0"}
    """

    items = {}

    for key, value in config.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(flatten_config(value, new_key, separator))
        else:
            items[new_key] = value

    return items


def config_to_dataframe(config: dict):
    """
    Convert a configuration dictionary to a pandas DataFrame.

    This is useful for displaying the final configuration inside notebooks.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: key, value.
    """

    import pandas as pd

    flat = flatten_config(config)

    return pd.DataFrame(
        [{"key": key, "value": value} for key, value in flat.items()]
    )


def validate_required_keys(config: dict, required_keys: list[str]) -> None:
    """
    Validate whether a configuration contains required nested keys.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    required_keys : list[str]
        List of required keys in dot format.
        Example: ["pipeline.backbone", "pipeline.classifier"]

    Raises
    ------
    KeyError
        If a required key is missing.
    """

    missing = []

    for key_path in required_keys:
        current = config
        parts = key_path.split(".")

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                missing.append(key_path)
                break
            current = current[part]

    if missing:
        raise KeyError(
            "Missing required configuration keys: "
            + ", ".join(missing)
        )


def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    key_path : str
        Dot-form key path.
        Example: "pipeline.backbone"
    default : Any
        Value returned if the key does not exist.

    Returns
    -------
    Any
        Configuration value.
    """

    current = config

    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def set_config_value(config: dict, key_path: str, value: Any) -> dict:
    """
    Set a nested configuration value using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    key_path : str
        Dot-form key path.
        Example: "pipeline.backbone"
    value : Any
        New value.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """

    result = deepcopy(config)
    current = result
    parts = key_path.split(".")

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value

    return result


def resolve_workspace(
    workspace: str | Path | None = None,
    project_name: str = "ml_image_classifier"
) -> Path:
    """
    Resolve a suitable workspace path for Colab, Kaggle, or local execution.

    Parameters
    ----------
    workspace : str | Path | None
        User-defined workspace path. If provided, it is used directly.
    project_name : str
        Default project folder name.

    Returns
    -------
    Path
        Resolved workspace path.
    """

    if workspace is not None:
        path = Path(workspace)
    elif Path("/kaggle/working").exists():
        path = Path("/kaggle/working") / project_name
    elif Path("/content").exists():
        path = Path("/content") / project_name
    else:
        path = Path.cwd() / project_name

    path.mkdir(parents=True, exist_ok=True)

    return path


def save_run_metadata(
    config: dict,
    output_dir: str | Path,
    filename: str = "run_metadata.json"
) -> Path:
    """
    Save configuration and environment metadata for reproducibility.

    Parameters
    ----------
    config : dict
        Final configuration dictionary.
    output_dir : str | Path
        Directory where metadata file will be saved.
    filename : str
        Metadata file name.

    Returns
    -------
    Path
        Saved metadata path.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": config,
        "environment": {
            "device": str(get_device()),
            "torch_available": torch is not None,
            "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
            "cuda_device_count": int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0,
        },
    }

    path = output_dir / filename
    save_json(metadata, path)

    return path