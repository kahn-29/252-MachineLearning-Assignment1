# modules/feature_extraction.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class ImagePathDataset(Dataset):
    """
    Dataset for feature extraction.

    It returns transformed image tensors and optionally labels.
    """

    def __init__(
        self,
        paths,
        labels=None,
        transform=None,
        fallback_size: int = 224,
    ):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels)
        self.transform = transform
        self.fallback_size = fallback_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new(
                "RGB",
                (self.fallback_size, self.fallback_size),
                (128, 128, 128),
            )

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img

        label = int(self.labels[idx])
        return img, label


def get_backbone(
    name: str,
    device=None,
    pretrained: bool = True,
    data_parallel: bool = False,
):
    """
    Return a pretrained CNN backbone without its classification head.

    Supported backbones:
    - resnet18
    - vgg16
    - efficientnet_b0

    Parameters
    ----------
    name : str
        Backbone name.
    device : torch.device | None
        Target device.
    pretrained : bool
        Whether to use pretrained ImageNet weights.
    data_parallel : bool
        Whether to wrap the model using nn.DataParallel.

    Returns
    -------
    torch.nn.Module
        Backbone model in eval mode.
    """

    name = name.lower().strip()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        model = nn.Sequential(*list(base.children())[:-1])

    elif name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        base = models.vgg16(weights=weights)
        model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    else:
        raise ValueError(
            f"Unsupported backbone: {name}. "
            "Supported backbones: resnet18, vgg16, efficientnet_b0."
        )

    if data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def get_feature_dim(backbone_name: str) -> int:
    """
    Return output feature dimension for a supported backbone.
    """

    backbone_name = backbone_name.lower().strip()

    feature_dims = {
        "resnet18": 512,
        "vgg16": 512,
        "efficientnet_b0": 1280,
    }

    if backbone_name not in feature_dims:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Available: {list(feature_dims.keys())}"
        )

    return feature_dims[backbone_name]


@torch.inference_mode()
def extract_features(
    df: pd.DataFrame,
    transform,
    backbone_name: str,
    batch_size: int = 128,
    device=None,
    num_workers: int = 0,
    path_col: str = "path",
    label_col: str = "label",
    pretrained: bool = True,
    data_parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract deep feature vectors from images using a pretrained backbone.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing image paths and labels.
    transform :
        Image transform.
    backbone_name : str
        Backbone name.
    batch_size : int
        Batch size for DataLoader.
    device : torch.device | None
        Target device.
    num_workers : int
        Number of DataLoader workers.
    path_col : str
        Column containing image paths.
    label_col : str
        Column containing labels.
    pretrained : bool
        Whether to use pretrained weights.
    data_parallel : bool
        Whether to use DataParallel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X, y
    """

    if path_col not in df.columns:
        raise KeyError(f"path_col not found in dataframe: {path_col}")

    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = df[path_col].values
    labels = df[label_col].values.astype(np.int64)

    dataset = ImagePathDataset(
        paths=paths,
        labels=labels,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = get_backbone(
        name=backbone_name,
        device=device,
        pretrained=pretrained,
        data_parallel=data_parallel,
    )

    features = []
    y_values = []

    for imgs, batch_labels in tqdm(
        loader,
        desc=f"Extracting features ({backbone_name})",
        leave=False,
    ):
        imgs = imgs.to(device, non_blocking=True)

        output = model(imgs)
        output = output.view(output.size(0), -1)

        features.append(output.detach().cpu().numpy())
        y_values.append(batch_labels.numpy())

    X = np.vstack(features)
    y = np.concatenate(y_values)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return X, y


def save_feature_split(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save feature matrix and label vector as .npy files.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    split_name : str
        Split name, e.g. train, val, test.
    output_dir : str | Path
        Output directory.

    Returns
    -------
    dict[str, Path]
        Saved file paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    return {
        "X": x_path,
        "y": y_path,
    }


def load_feature_split(
    split_name: str,
    feature_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature matrix and label vector for one split.

    Parameters
    ----------
    split_name : str
        Split name, e.g. train, val, test.
    feature_dir : str | Path
        Feature directory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X, y
    """

    feature_dir = Path(feature_dir)

    x_path = feature_dir / f"X_{split_name}.npy"
    y_path = feature_dir / f"y_{split_name}.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    return X, y


def feature_files_exist(
    feature_dir: str | Path,
    split_names=("train", "val", "test"),
) -> bool:
    """
    Check whether feature files already exist for all requested splits.
    """

    feature_dir = Path(feature_dir)

    for split in split_names:
        if not (feature_dir / f"X_{split}.npy").exists():
            return False
        if not (feature_dir / f"y_{split}.npy").exists():
            return False

    return True


def extract_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    eval_transform,
    backbone_name: str,
    batch_size: int = 128,
    device=None,
    num_workers: int = 0,
    output_dir: str | Path | None = None,
    pretrained: bool = True,
    data_parallel: bool = False,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """
    Extract features for train, validation, and test splits.

    Training transform may include augmentation.
    Validation and test transforms should be deterministic.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    val_df : pd.DataFrame
        Validation dataframe.
    test_df : pd.DataFrame
        Test dataframe.
    train_transform :
        Transform applied to training images.
    eval_transform :
        Transform applied to validation and test images.
    backbone_name : str
        Backbone name.
    batch_size : int
        Batch size.
    device : torch.device | None
        Target device.
    num_workers : int
        Number of DataLoader workers.
    output_dir : str | Path | None
        If provided, save feature files to this directory.
    pretrained : bool
        Whether to use pretrained weights.
    data_parallel : bool
        Whether to use DataParallel.
    force_recompute : bool
        If False and output_dir contains feature files, load existing files.

    Returns
    -------
    dict
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test.
    """

    if output_dir is not None:
        output_dir = Path(output_dir)

        if (
            not force_recompute
            and feature_files_exist(output_dir, split_names=("train", "val", "test"))
        ):
            X_train, y_train = load_feature_split("train", output_dir)
            X_val, y_val = load_feature_split("val", output_dir)
            X_test, y_test = load_feature_split("test", output_dir)

            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "loaded_from_cache": True,
                "feature_dir": output_dir,
            }

    X_train, y_train = extract_features(
        df=train_df,
        transform=train_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pretrained=pretrained,
        data_parallel=data_parallel,
    )

    X_val, y_val = extract_features(
        df=val_df,
        transform=eval_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pretrained=pretrained,
        data_parallel=data_parallel,
    )

    X_test, y_test = extract_features(
        df=test_df,
        transform=eval_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pretrained=pretrained,
        data_parallel=data_parallel,
    )

    if output_dir is not None:
        save_feature_split(X_train, y_train, "train", output_dir)
        save_feature_split(X_val, y_val, "val", output_dir)
        save_feature_split(X_test, y_test, "test", output_dir)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "loaded_from_cache": False,
        "feature_dir": output_dir,
    }


def summarize_feature_matrix(
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Return basic information about a feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray | None
        Optional label vector.

    Returns
    -------
    dict
        Feature summary.
    """

    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]) if X.ndim == 2 else None,
        "shape": tuple(X.shape),
        "dtype": str(X.dtype),
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
    }

    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        summary["label_distribution"] = {
            int(k): int(v) for k, v in zip(unique, counts)
        }

    return summary


def save_feature_metadata(
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save feature extraction metadata as JSON.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary.
    output_path : str | Path
        Output JSON path.

    Returns
    -------
    Path
        Saved metadata path.
    """

    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_metadata = {}

    for key, value in metadata.items():
        if isinstance(value, Path):
            serializable_metadata[key] = str(value)
        elif isinstance(value, np.ndarray):
            serializable_metadata[key] = value.tolist()
        else:
            serializable_metadata[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metadata, f, indent=4, ensure_ascii=False)

    return output_path


def build_feature_dir_name(
    preprocessing: str,
    backbone: str,
    image_size: int,
) -> str:
    """
    Build a standardized feature directory name.

    Example
    -------
    augmented_efficientnet_b0_224
    """

    preprocessing = preprocessing.lower().replace(" ", "_")
    backbone = backbone.lower().replace(" ", "_")

    return f"{preprocessing}_{backbone}_{image_size}"


def load_or_extract_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    eval_transform,
    backbone_name: str,
    feature_root: str | Path,
    preprocessing_name: str,
    image_size: int = 224,
    batch_size: int = 128,
    device=None,
    num_workers: int = 0,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """
    Convenience wrapper that creates a standardized feature directory and
    loads or extracts train/val/test features.
    """

    feature_root = Path(feature_root)
    feature_dir = feature_root / build_feature_dir_name(
        preprocessing=preprocessing_name,
        backbone=backbone_name,
        image_size=image_size,
    )

    result = extract_feature_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_transform=train_transform,
        eval_transform=eval_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        output_dir=feature_dir,
        force_recompute=force_recompute,
    )

    metadata = {
        "preprocessing": preprocessing_name,
        "backbone": backbone_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "feature_dim": get_feature_dim(backbone_name),
        "train_shape": result["X_train"].shape,
        "val_shape": result["X_val"].shape,
        "test_shape": result["X_test"].shape,
        "loaded_from_cache": result["loaded_from_cache"],
    }

    save_feature_metadata(metadata, feature_dir / "feature_metadata.json")
    result["metadata"] = metadata

    return result