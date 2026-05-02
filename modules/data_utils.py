# modules/data_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


def _normalize_extensions(extensions: Iterable[str] | None) -> list[str]:
    if extensions is None:
        return DEFAULT_IMAGE_EXTENSIONS

    patterns: list[str] = []

    for ext in extensions:
        token = str(ext).strip().lower()
        if not token:
            continue
        if token.startswith("*."):
            patterns.append(token)
        elif token.startswith("."):
            patterns.append(f"*{token}")
        elif token.startswith("*"):
            patterns.append(token)
        else:
            patterns.append(f"*.{token}")

    return patterns or DEFAULT_IMAGE_EXTENSIONS


def _contains_images(root: Path) -> bool:
    for pattern in DEFAULT_IMAGE_EXTENSIONS:
        if any(root.rglob(pattern)):
            return True
    return False


def _download_kaggle_dataset(dataset_id: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download datasets. Install it with: pip install kagglehub"
        ) from exc

    dataset_path = Path(kagglehub.dataset_download(dataset_id))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Downloaded dataset path does not exist: {dataset_path}")
    return dataset_path


def _default_class_map() -> dict[str, tuple[int, str]]:
    return {
        "cat": (0, "Cat"),
        "cats": (0, "Cat"),
        "dog": (1, "Dog"),
        "dogs": (1, "Dog"),
    }


def resolve_dataset_root(
    dataset_id: str | None = None,
    local_root: str | Path | None = None,
    kaggle_input_dir: str | Path = "/kaggle/input"
) -> Path:
    """Resolve dataset root from local path, Kaggle input, or kagglehub download."""

    if local_root is not None:
        local_root = Path(local_root)

        if not local_root.exists():
            raise FileNotFoundError(f"local_root does not exist: {local_root}")

        if not _contains_images(local_root):
            raise ValueError(f"local_root has no supported image files: {local_root}")

        return local_root.resolve()

    kaggle_input_dir = Path(kaggle_input_dir)

    if kaggle_input_dir.exists():
        if _contains_images(kaggle_input_dir):
            return kaggle_input_dir.resolve()

    if dataset_id is not None:
        downloaded_root = _download_kaggle_dataset(dataset_id)
        if _contains_images(downloaded_root):
            return downloaded_root.resolve()
        raise ValueError(f"Downloaded dataset has no supported image files: {downloaded_root}")

    raise ValueError("Cannot resolve dataset root. Provide local_root or a valid dataset_id.")


def list_image_paths(
    root: str | Path,
    extensions: Iterable[str] | None = None
) -> list[str]:
    """Recursively list supported image files under root."""

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    patterns = _normalize_extensions(extensions)
    paths = []

    for pattern in patterns:
        paths.extend(root.rglob(pattern))

    return sorted(str(p.resolve()) for p in paths if p.is_file())


def infer_label_from_path(
    path: str | Path,
    class_map: dict | None = None,
) -> tuple[int | None, str | None]:
    """Infer numeric label and class name from folder/file tokens."""

    path = Path(path)
    mapping = _default_class_map() if class_map is None else class_map

    filename = path.name.lower().replace("-", "_")
    parent = path.parent.name.lower().replace("-", "_")
    parts = [part.lower() for part in path.parts]

    if parent in mapping:
        label, label_name = mapping[parent]
        return int(label), str(label_name)

    for part in reversed(parts):
        token = part.replace("-", "_")
        if token in mapping:
            label, label_name = mapping[token]
            return int(label), str(label_name)

    stem_tokens = filename.split("_") + filename.split(".")
    for token in stem_tokens:
        token = token.strip()
        if token in mapping:
            label, label_name = mapping[token]
            return int(label), str(label_name)

    return None, None


def build_raw_dataframe(
    root: str | Path,
    extensions: Iterable[str] | None = None,
    class_map: dict | None = None,
    drop_unknown: bool = True
) -> pd.DataFrame:
    """Build a raw dataframe with columns path, label, label_name."""

    paths = list_image_paths(root, extensions=extensions)

    records = []

    for path in paths:
        label, label_name = infer_label_from_path(path, class_map=class_map)

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
        raise ValueError("No images found for the selected root/extensions.")

    df = df.reset_index(drop=True)

    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified train/val/test splits with reset indices."""

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
    """Summarize class counts and percentages for a dataframe."""

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
    """Summarize class distribution for train, validation, and test splits."""

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


def sample_by_class(
    df: pd.DataFrame,
    n_per_class: int = 5,
    label_col: str = "label_name",
    seed: int = 42
) -> pd.DataFrame:
    """Return a balanced per-class sample for visualization or quick EDA."""

    samples = []

    for _, group in df.groupby(label_col):
        n = min(n_per_class, len(group))
        samples.append(group.sample(n=n, random_state=seed))

    return pd.concat(samples, ignore_index=True)
