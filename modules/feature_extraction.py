from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    """
    Dataset that returns transformed image only.
    Used for feature extraction.
    """
    def __init__(self, paths, transform):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...


def get_backbone(
    name: str,
    device,
    pretrained: bool = True
):
    """
    Return pretrained backbone without classification head.
    Supported:
    resnet18, vgg16, efficientnet_b0.
    """
    ...


def get_feature_dim(backbone_name: str) -> int:
    """
    Return output feature dimension of selected backbone.
    Example:
    ResNet18 -> 512
    VGG16 -> 512
    EfficientNet-B0 -> 1280
    """
    ...


def extract_features(
    df: pd.DataFrame,
    transform,
    backbone_name: str,
    batch_size: int,
    device,
    num_workers: int = 0,
    path_col: str = "path",
    label_col: str = "label"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label vector y from dataframe.
    """
    ...


def extract_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transform,
    backbone_name: str,
    batch_size: int,
    device,
    output_dir: str | Path | None = None
) -> dict:
    """
    Extract features for train/val/test.
    Optionally save to output_dir.
    Return dictionary containing X_train, y_train, X_val, y_val, X_test, y_test.
    """
    ...


def save_feature_split(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    output_dir: str | Path
) -> None:
    """
    Save X and y as .npy files.
    """
    ...


def load_feature_split(
    split_name: str,
    feature_dir: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X and y for one split.
    """
    ...


def feature_files_exist(
    feature_dir: str | Path,
    split_names=("train", "val", "test")
) -> bool:
    """
    Check whether feature files already exist.
    """
    ...