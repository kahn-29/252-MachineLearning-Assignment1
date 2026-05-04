"""
Configuration and runtime utilities for the cat/dog image-classification project.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

try:  # Keep config utilities importable even in lightweight environments.
    import torch
except ImportError:  # pragma: no cover - depends on runtime environment.
    torch = None  # type: ignore[assignment]


PROJECT_NAME = "ml_image_classifier"

DEFAULT_SEED = 42

SUPPORTED_PREPROCESSING_MODES = ("stretch", "letterbox", "center_crop", "augmented")
SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet18", "vgg16")
SUPPORTED_CLASSIFIERS = (
    "logistic_regression",
    "svm_linear",
    "random_forest",
    "voting_soft",
    "stacking",
)


# -----------------------------------------------------------------------------
# Reproducibility and runtime helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def get_device() -> Any:
    """Return the best available PyTorch device, or the string ``"cpu"`` without PyTorch."""
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs: str | Path | None) -> None:
    """Create all provided directories if they do not already exist.

    ``None`` values are ignored so callers can pass optional paths safely.
    """
    for directory in dirs:
        if directory is None:
            continue
        Path(directory).expanduser().mkdir(parents=True, exist_ok=True)


def resolve_workspace(
    workspace: str | Path | None = None,
    project_name: str = PROJECT_NAME,
) -> Path:
    """Resolve a stable workspace path across local, Colab, and Kaggle runs.

    Resolution order:
    1. explicit ``workspace`` argument,
    2. ``ML_PROJECT_WORKSPACE`` environment variable,
    3. ``/content/<project_name>`` on Google Colab-like runtimes,
    4. ``/kaggle/working/<project_name>`` on Kaggle notebooks,
    5. ``./<project_name>`` in local runs.

    The directory is not created here; call :func:`ensure_dirs` for side effects.
    """
    if workspace is not None:
        return Path(workspace).expanduser().resolve()

    env_workspace = os.environ.get("ML_PROJECT_WORKSPACE")
    if env_workspace:
        return Path(env_workspace).expanduser().resolve()

    if Path("/content").exists():
        return Path("/content") / project_name

    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / project_name

    return (Path.cwd() / project_name).resolve()


# -----------------------------------------------------------------------------
# Configuration handling
# -----------------------------------------------------------------------------


def deep_update(
    default: Mapping[str, Any],
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deep-merged config without mutating ``default``.

    Nested dictionaries are merged recursively. Non-dictionary values in
    ``override`` replace values from ``default``.
    """
    result: dict[str, Any] = copy.deepcopy(dict(default))
    if not override:
        return result

    def _merge(base: MutableMapping[str, Any], patch: Mapping[str, Any]) -> MutableMapping[str, Any]:
        for key, value in patch.items():
            if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
                _merge(base[key], value)  # type: ignore[arg-type]
            else:
                base[key] = copy.deepcopy(value)
        return base

    return dict(_merge(result, override))


def get_default_config() -> dict[str, Any]:
    """Return the project's default configuration as a fresh deep copy."""
    workspace = resolve_workspace(project_name=PROJECT_NAME)

    config: dict[str, Any] = {
        "project": {
            "name": PROJECT_NAME,
            "task": "binary_image_classification",
            "target": "cat_vs_dog",
        },
        "seed": DEFAULT_SEED,
        "runtime": {
            "deterministic": False,
            "device": "auto",
            "num_workers": 0,
        },
        "paths": {
            "workspace": str(workspace),
            "data_dir": str(workspace / "data"),
            "raw_data_dir": str(workspace / "data" / "raw"),
            "processed_data_dir": str(workspace / "data" / "processed"),
            "features_dir": str(workspace / "features"),
            "models_dir": str(workspace / "models"),
            "reports_dir": str(workspace / "reports"),
            "figures_dir": str(workspace / "reports" / "figures"),
            "results_dir": str(workspace / "reports" / "results"),
        },
        "dataset": {
            "dataset_id": "tongpython/cat-and-dog",
            "local_root": None,
            "kaggle_input_dir": "/kaggle/input",
            "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            "class_map": {
                "cat": 0,
                "cats": 0,
                "dog": 1,
                "dogs": 1,
            },
            "label_names": {
                0: "cat",
                1: "dog",
            },
            "drop_unknown": True,
        },
        "split": {
            "train_ratio": 0.80,
            "val_ratio": 0.10,
            "test_ratio": 0.10,
            "label_col": "label",
        },
        "audit": {
            "compute_hash": True,
            "path_col": "path",
            "label_col": "label",
            "label_name_col": "label_name",
        },
        "cleaning": {
            "enabled": False,
            "remove_corrupted": True,
            "remove_duplicates": True,
            "duplicate_hamming_threshold": 4,
            "min_side": 64,
            "max_aspect_extremity": 5.0,
            "min_blur_laplacian": 40.0,
            "min_entropy": 0.0,
            "max_near_mono_ratio": 0.92,
            "max_dark_ratio": 0.98,
            "max_bright_ratio": 0.98,
            "min_mean_sat": 0.0,
            "min_chroma_mean": 0.0,
            "min_center_saliency_ratio": 0.0,
            # None means "disabled"; cleaning.py will fall back to np.inf.
            "max_compression_artifact": None,
        },
        "preprocessing": {
            "mode": "augmented",
            "image_size": 224,
            "train_augmentation": False,
            "normalize": "imagenet",
        },
        "feature_extraction": {
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "batch_size": 128,
            "num_workers": 2,
            "data_parallel": False,
            "force_recompute": False,
            "file_format": "npy",
            "on_error": "raise",
        },
        "classifier": {
            "name": "voting_soft",
            "params": {
                "voting": "soft",
                "n_jobs": -1,
                "lr": {
                    "C": 0.1,
                    "max_iter": 3000,
                    "n_jobs": -1,
                },
                "svm": {
                    "C": 0.1,
                    "probability": True,
                },
                "rf": {
                    "n_estimators": 100,
                    "n_jobs": -1,
                },
            },
        },
        "grid_search": {
            "primary_metric": "f1_macro",
            "tie_breakers": ["accuracy"],
            "search_space": {
                "preprocessing.mode": ["stretch", "letterbox", "center_crop"],
                "feature_extraction.backbone": ["efficientnet_b0", "resnet18"],
                "classifier.name": ["logistic_regression", "svm_linear", "random_forest"],
            },
        },
        "deep_learning": {
            "enabled": False,
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "image_size": 224,
            "batch_size": 32,
            "head_epochs": 3,
            "fine_tune_epochs": 2,
            "head_lr": 1e-3,
            "fine_tune_lr": 1e-5,
            "dropout": 0.2,
            "unfreeze_last_blocks": 1,
        },
    }

    return copy.deepcopy(config)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate required config sections and common value ranges."""
    required_sections = (
        "project",
        "seed",
        "paths",
        "dataset",
        "split",
        "preprocessing",
        "feature_extraction",
        "classifier",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError(f"Missing config section(s): {missing}")

    _validate_split_config(config["split"])
    _validate_preprocessing_config(config["preprocessing"])
    _validate_feature_extraction_config(config["feature_extraction"])
    _validate_classifier_config(config["classifier"])
    _validate_runtime_config(config.get("runtime", {}))
    _validate_cleaning_config(config.get("cleaning", {}))

    if "deep_learning" in config:
        _validate_deep_learning_config(config["deep_learning"])


def _validate_split_config(split: Mapping[str, Any]) -> None:
    ratios = [
        float(split.get("train_ratio", 0.0)),
        float(split.get("val_ratio", 0.0)),
        float(split.get("test_ratio", 0.0)),
    ]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}")

    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} with sum={ratio_sum:.6f}")


def _validate_preprocessing_config(preprocessing: Mapping[str, Any]) -> None:
    mode = str(preprocessing.get("mode", "")).lower().strip()
    if mode not in SUPPORTED_PREPROCESSING_MODES:
        raise ValueError(
            f"Unsupported preprocessing.mode: {mode}. "
            f"Supported modes: {list(SUPPORTED_PREPROCESSING_MODES)}"
        )

    _validate_image_size(preprocessing.get("image_size"), field_name="preprocessing.image_size")


def _validate_feature_extraction_config(feature_extraction: Mapping[str, Any]) -> None:
    backbone = str(feature_extraction.get("backbone", "")).lower().strip()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported feature_extraction.backbone: {backbone}. "
            f"Supported backbones: {list(SUPPORTED_BACKBONES)}"
        )

    batch_size = int(feature_extraction.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError("feature_extraction.batch_size must be positive")

    num_workers = int(feature_extraction.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("feature_extraction.num_workers must be non-negative")

    file_format = str(feature_extraction.get("file_format", "npy")).lower().strip()
    if file_format not in {"npy", "h5"}:
        raise ValueError("feature_extraction.file_format must be either 'npy' or 'h5'")


def _validate_classifier_config(classifier: Mapping[str, Any]) -> None:
    name = str(classifier.get("name", "")).lower().strip()
    if name not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unsupported classifier.name: {name}. "
            f"Supported classifiers: {list(SUPPORTED_CLASSIFIERS)}"
        )


def _validate_runtime_config(runtime: Mapping[str, Any]) -> None:
    num_workers = int(runtime.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("runtime.num_workers must be non-negative")


def _validate_cleaning_config(cleaning: Mapping[str, Any]) -> None:
    duplicate_threshold = int(cleaning.get("duplicate_hamming_threshold", 0))
    if duplicate_threshold < 0:
        raise ValueError("cleaning.duplicate_hamming_threshold must be non-negative")

    for key in [
        "min_side",
        "max_aspect_extremity",
        "min_blur_laplacian",
        "min_entropy",
        "max_near_mono_ratio",
        "max_dark_ratio",
        "max_bright_ratio",
        "min_mean_sat",
        "min_chroma_mean",
        "min_center_saliency_ratio",
    ]:
        value = cleaning.get(key)
        if value is not None and float(value) < 0:
            raise ValueError(f"cleaning.{key} must be non-negative")


def _validate_deep_learning_config(deep_learning: Mapping[str, Any]) -> None:
    backbone = str(deep_learning.get("backbone", "")).lower().strip()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported deep_learning.backbone: {backbone}. "
            f"Supported backbones: {list(SUPPORTED_BACKBONES)}"
        )

    _validate_image_size(deep_learning.get("image_size"), field_name="deep_learning.image_size")

    for key in ["batch_size", "head_epochs", "fine_tune_epochs"]:
        value = int(deep_learning.get(key, 0))
        if value < 0 if key.endswith("epochs") else value <= 0:
            raise ValueError(f"deep_learning.{key} has an invalid value: {value}")

    dropout = float(deep_learning.get("dropout", 0.0))
    if not 0 <= dropout < 1:
        raise ValueError("deep_learning.dropout must be in [0, 1)")

    for key in ["head_lr", "fine_tune_lr"]:
        value = float(deep_learning.get(key, 0.0))
        if value <= 0:
            raise ValueError(f"deep_learning.{key} must be positive")


def _validate_image_size(value: Any, field_name: str) -> None:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        return

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{field_name} tuple/list must have length 2")
        if any(int(size) <= 0 for size in value):
            raise ValueError(f"{field_name} tuple/list must contain positive integers")
        return

    raise ValueError(f"{field_name} must be an int or a length-2 tuple/list")


def config_to_run_name(config: Mapping[str, Any]) -> str:
    """Build a compact, filesystem-safe run name from the main config choices."""
    parts = [
        str(config.get("preprocessing", {}).get("mode", "preprocess")),
        str(config.get("feature_extraction", {}).get("backbone", "backbone")),
        str(config.get("classifier", {}).get("name", "classifier")),
    ]
    raw_name = "__".join(parts).lower()
    safe_chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in raw_name]
    return "".join(safe_chars).strip("_")


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary as pretty JSON."""
    path = Path(path)
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(config), file, ensure_ascii=False, indent=2, allow_nan=False)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a configuration dictionary from JSON."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON objects into strict JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "PROJECT_NAME",
    "DEFAULT_SEED",
    "SUPPORTED_PREPROCESSING_MODES",
    "SUPPORTED_BACKBONES",
    "SUPPORTED_CLASSIFIERS",
    "set_seed",
    "get_device",
    "ensure_dirs",
    "resolve_workspace",
    "deep_update",
    "get_default_config",
    "validate_config",
    "config_to_run_name",
    "save_config",
    "load_config",
]
