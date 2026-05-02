from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def download_kaggle_dataset(dataset_id: str) -> Path:
    """
    Download public Kaggle dataset using kagglehub.
    Return dataset root path.
    """
    ...


def resolve_dataset_root(dataset_id: str, local_root: str | None = None) -> Path:
    """
    Resolve dataset path.
    Priority:
    1. local_root if provided
    2. /kaggle/input if available
    3. kagglehub download
    """
    ...


def list_image_paths(root: str | Path, extensions=None) -> list[str]:
    """
    Recursively list image paths under root.
    Default extensions: jpg, jpeg, png.
    """
    ...


def infer_label_from_path(path: str | Path) -> tuple[int | None, str | None]:
    """
    Infer label from folder name or file name.
    Return:
    0, "Cat" for cat
    1, "Dog" for dog
    None, None if label cannot be inferred.
    """
    ...


def build_raw_dataframe(root: str | Path, extensions=None) -> pd.DataFrame:
    """
    Build dataframe with columns:
    path, label, label_name.
    """
    ...


def verify_image_readable(path: str | Path) -> bool:
    """
    Check whether an image can be opened by PIL.
    """
    ...


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/validation/test split.
    Return train_df, val_df, test_df.
    """
    ...


def summarize_class_distribution(df: pd.DataFrame, label_col: str = "label_name") -> pd.DataFrame:
    """
    Return count and percentage of each class.
    """
    ...