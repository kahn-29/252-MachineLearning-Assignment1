# modules/config_utils.py

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def deep_update(default: dict, override: dict | None) -> dict:
    """Return a recursively merged copy of default overridden by override."""

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


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for random, numpy, and torch when available."""

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
    """Return torch.device('cuda'|'cpu') or 'cpu' when torch is unavailable."""

    if torch is None:
        return "cpu"

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs: str | Path) -> None:
    """Create one or more directories if they do not already exist."""

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def resolve_workspace(
    workspace: str | Path | None = None,
    project_name: str = "ml_image_classifier"
) -> Path:
    """Resolve and create a workspace path for local, Kaggle, or Colab runs."""

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