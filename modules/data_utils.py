# modules/data_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


DEFAULT_IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


def download_kaggle_dataset(dataset_id: str) -> Path:
    """
    Download a public Kaggle dataset using kagglehub.

    Parameters
    ----------
    dataset_id : str
        Kaggle dataset ID.
        Example: "tongpython/cat-and-dog"

    Returns
    -------
    Path
        Local path to the downloaded dataset.
    """

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download Kaggle datasets. "
            "Install it with: pip install kagglehub"
        ) from exc

    path = kagglehub.dataset_download(dataset_id)
    return Path(path)


def resolve_dataset_root(
    dataset_id: str | None = None,
    local_root: str | Path | None = None,
    kaggle_input_dir: str | Path = "/kaggle/input"
) -> Path:
    """
    Resolve dataset root path for local, Kaggle, or Colab execution.

    Priority
    --------
    1. Use local_root if provided.
    2. Search /kaggle/input if available and contains images.
    3. Download dataset using kagglehub if dataset_id is provided.

    Parameters
    ----------
    dataset_id : str | None
        Kaggle dataset ID.
    local_root : str | Path | None
        Manually provided dataset root.
    kaggle_input_dir : str | Path
        Kaggle input directory.

    Returns
    -------
    Path
        Resolved dataset root path.
    """

    if local_root is not None:
        local_root = Path(local_root)

        if not local_root.exists():
            raise FileNotFoundError(f"local_root does not exist: {local_root}")

        return local_root

    kaggle_input_dir = Path(kaggle_input_dir)

    if kaggle_input_dir.exists():
        image_paths = list_image_paths(kaggle_input_dir)

        if len(image_paths) > 0:
            return kaggle_input_dir

    if dataset_id is not None:
        return download_kaggle_dataset(dataset_id)

    raise ValueError(
        "Cannot resolve dataset root. Provide either local_root or dataset_id."
    )


def normalize_extensions(extensions: Iterable[str] | None = None) -> list[str]:
    """
    Normalize image extension patterns for recursive glob search.

    Parameters
    ----------
    extensions : Iterable[str] | None
        Extension patterns or suffixes.
        Examples:
        ["*.jpg", "*.png"] or ["jpg", "png"]

    Returns
    -------
    list[str]
        Normalized glob patterns.
    """

    if extensions is None:
        return DEFAULT_IMAGE_EXTENSIONS.copy()

    normalized = []

    for ext in extensions:
        ext = str(ext).strip().lower()

        if not ext:
            continue

        if ext.startswith("*."):
            normalized.append(ext)
        elif ext.startswith("."):
            normalized.append(f"*{ext}")
        else:
            normalized.append(f"*.{ext}")

    return normalized


def list_image_paths(
    root: str | Path,
    extensions: Iterable[str] | None = None
) -> list[str]:
    """
    Recursively list image paths under a root directory.

    Parameters
    ----------
    root : str | Path
        Dataset root directory.
    extensions : Iterable[str] | None
        Image extensions to include.

    Returns
    -------
    list[str]
        Sorted list of image file paths as strings.
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    patterns = normalize_extensions(extensions)
    paths = []

    for pattern in patterns:
        paths.extend(root.rglob(pattern))

    return sorted([str(p) for p in paths if p.is_file()])


def infer_label_from_path(path: str | Path) -> tuple[int | None, str | None]:
    """
    Infer label from image path.

    For the Cat-Dog dataset:
    - Cat is encoded as 0.
    - Dog is encoded as 1.

    The function prioritizes folder names, then falls back to file names.

    Parameters
    ----------
    path : str | Path
        Image path.

    Returns
    -------
    tuple[int | None, str | None]
        (label, label_name)
        Returns (None, None) if label cannot be inferred.
    """

    path = Path(path)

    filename = path.name.lower()
    parent = path.parent.name.lower()
    parts = [part.lower() for part in path.parts]

    cat_tokens = {"cat", "cats"}
    dog_tokens = {"dog", "dogs"}

    if parent in cat_tokens:
        return 0, "Cat"

    if parent in dog_tokens:
        return 1, "Dog"

    if any(part in cat_tokens for part in parts):
        return 0, "Cat"

    if any(part in dog_tokens for part in parts):
        return 1, "Dog"

    if filename.startswith("cat"):
        return 0, "Cat"

    if filename.startswith("dog"):
        return 1, "Dog"

    return None, None


def build_raw_dataframe(
    root: str | Path,
    extensions: Iterable[str] | None = None,
    drop_unknown: bool = True
) -> pd.DataFrame:
    """
    Build a raw image dataframe from a dataset directory.

    Parameters
    ----------
    root : str | Path
        Dataset root directory.
    extensions : Iterable[str] | None
        Image extensions to include.
    drop_unknown : bool
        If True, remove images whose labels cannot be inferred.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        path, label, label_name.
    """

    paths = list_image_paths(root, extensions=extensions)

    records = []

    for path in paths:
        label, label_name = infer_label_from_path(path)

        if label is None and drop_unknown:
            continue

        records.append(
            {
                "path": str(path),
                "label": label,
                "label_name": label_name,
            }
        )

    df = pd.DataFrame(records)

    if len(df) == 0:
        raise ValueError(
            "No labeled images found. Check dataset structure and label inference."
        )

    df = df.reset_index(drop=True)

    return df


def verify_image_readable(path: str | Path) -> bool:
    """
    Check whether an image can be opened by PIL.

    Parameters
    ----------
    path : str | Path
        Image path.

    Returns
    -------
    bool
        True if the image is readable, otherwise False.
    """

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def add_readability_flag(
    df: pd.DataFrame,
    path_col: str = "path"
) -> pd.DataFrame:
    """
    Add an is_readable column to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    path_col : str
        Column containing image paths.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional is_readable column.
    """

    output = df.copy()
    output["is_readable"] = output[path_col].apply(verify_image_readable)

    return output


def filter_readable_images(
    df: pd.DataFrame,
    path_col: str = "path"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into readable and unreadable images.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    path_col : str
        Column containing image paths.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        readable_df, unreadable_df
    """

    checked_df = add_readability_flag(df, path_col=path_col)

    readable_df = checked_df[checked_df["is_readable"]].copy()
    unreadable_df = checked_df[~checked_df["is_readable"]].copy()

    readable_df = readable_df.reset_index(drop=True)
    unreadable_df = unreadable_df.reset_index(drop=True)

    return readable_df, unreadable_df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train, validation, and test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    train_ratio : float
        Ratio of training data.
    val_ratio : float
        Ratio of validation data.
    test_ratio : float
        Ratio of test data.
    seed : int
        Random seed.
    label_col : str
        Label column for stratification.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """

    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {total}."
        )

    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    if df[label_col].nunique() < 2:
        raise ValueError("Stratified split requires at least two classes.")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[label_col],
        random_state=seed,
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df[label_col],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def summarize_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label_name"
) -> pd.DataFrame:
    """
    Summarize class counts and percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Class label column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        class, count, percentage.
    """

    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()

    summary = pd.DataFrame(
        {
            "class": counts.index,
            "count": counts.values,
            "percentage": counts.values / total * 100,
        }
    )

    return summary


def summarize_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label_name"
) -> pd.DataFrame:
    """
    Summarize class distribution across train/validation/test splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    val_df : pd.DataFrame
        Validation dataframe.
    test_df : pd.DataFrame
        Testing dataframe.
    label_col : str
        Class label column.

    Returns
    -------
    pd.DataFrame
        Distribution table with split, class, count, and percentage.
    """

    records = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        summary = summarize_class_distribution(split_df, label_col=label_col)
        summary.insert(0, "split", split_name)
        records.append(summary)

    return pd.concat(records, ignore_index=True)


def add_image_basename(
    df: pd.DataFrame,
    path_col: str = "path"
) -> pd.DataFrame:
    """
    Add file basename column for display/debugging.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    path_col : str
        Column containing image paths.

    Returns
    -------
    pd.DataFrame
        DataFrame with basename column.
    """

    output = df.copy()
    output["basename"] = output[path_col].apply(lambda x: Path(x).name)

    return output


def sample_by_class(
    df: pd.DataFrame,
    n_per_class: int = 5,
    label_col: str = "label_name",
    seed: int = 42
) -> pd.DataFrame:
    """
    Sample a fixed number of images per class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    n_per_class : int
        Number of images to sample from each class.
    label_col : str
        Class label column.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe.
    """

    samples = []

    for _, group in df.groupby(label_col):
        n = min(n_per_class, len(group))
        samples.append(group.sample(n=n, random_state=seed))

    return pd.concat(samples, ignore_index=True)


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False
) -> None:
    """
    Save dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    path : str | Path
        Output path.
    index : bool
        Whether to save dataframe index.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """
    Load dataframe from CSV.

    Parameters
    ----------
    path : str | Path
        CSV path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def build_or_load_raw_dataframe(
    root: str | Path,
    cache_path: str | Path | None = None,
    extensions: Iterable[str] | None = None,
    drop_unknown: bool = True
) -> pd.DataFrame:
    """
    Build raw dataframe or load it from cache if available.

    Parameters
    ----------
    root : str | Path
        Dataset root.
    cache_path : str | Path | None
        Optional CSV cache path.
    extensions : Iterable[str] | None
        Image extensions.
    drop_unknown : bool
        Whether to drop images with unknown labels.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """

    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists():
            return pd.read_csv(cache_path)

    df = build_raw_dataframe(
        root=root,
        extensions=extensions,
        drop_unknown=drop_unknown,
    )

    if cache_path is not None:
        save_dataframe(df, cache_path)

    return df