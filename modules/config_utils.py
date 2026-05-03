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



def get_default_config() -> dict:
    return {
        "DATASET": {
            "kaggle_id": "tongpython/cat-and-dog",
        },
        "CLEANING": {
            "blur_laplacian_min":   40.0,
            "min_side_min":         64,
            "near_mono_ratio_max":  0.92,
            "aspect_extremity_max": 5.0,
            "remove_corrupted":     True,
            "remove_duplicates":    False,   # set True to enable phash near-duplicate removal
        },
        "SPLIT": {
            "train": 0.8,
            "val":   0.1,
            "test":  0.1,
            "seed":  42,
        },
        "CLASSICAL": {
            # Feature extractor: "resnet18" | "vgg16" | "efficientnet_b0"
            "backbone":     "efficientnet_b0",
            # Resize mode: "letterbox" | "stretch" | "center_crop"
            "preprocessing": "letterbox",
            "image_size":    224,
            "batch_size":    64,
            # Classifier: "logistic_regression" | "svm_linear" | "random_forest"
            "classifier":   "logistic_regression",
            # Ensemble: when True, trains all 3 base + Voting + Stacking
            "ENSEMBLE":     False,
            # Classifier hyperparameters
            "lr_C":          1.0,
            "svm_C":         1.0,
            "rf_n_estimators": 200,
            "rf_max_depth":  None,
        },
    }


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