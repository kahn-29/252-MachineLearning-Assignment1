# modules/feature_extraction.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


class ImagePathDataset(Dataset):
    """
    Dataset used for extracting image features from file paths.
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
    Return a pretrained CNN backbone without the classification head.

    Supported backbones:
    - resnet18
    - vgg16
    - efficientnet_b0
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
    Return feature dimension of a supported backbone.
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
            f"Available backbones: {list(feature_dims.keys())}"
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
    Extract feature matrix X and label vector y from an image dataframe.
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
    Load feature matrix and label vector from .npy files.
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
    Check whether feature files exist for all requested splits.
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
    Extract or load features for train, validation, and test splits.

    If output_dir is provided and cached feature files already exist,
    the function loads them unless force_recompute=True.
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