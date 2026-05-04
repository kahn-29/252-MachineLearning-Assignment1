from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save JSON data, converting common ML types into JSON-safe values."""
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save a dataframe to .csv or .parquet."""
    path = Path(path)
    _ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported dataframe format: {suffix}")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from .csv or .parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe format: {suffix}")


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to .npy, creating parent folders if needed."""
    path = Path(path)
    _ensure_parent(path)
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from .npy without allowing pickles."""
    return np.load(Path(path), allow_pickle=False)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save a trusted Python object to pickle."""
    path = Path(path)
    _ensure_parent(path)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> Any:
    """Load a trusted pickle file only."""
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_feature_split(X: np.ndarray, y: np.ndarray, split_name: str, output_dir: str | Path) -> dict[str, Path]:
    """Save a feature split using the X_<split>.npy / y_<split>.npy convention."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"
    save_numpy(X, x_path)
    save_numpy(y, y_path)
    return {"X": x_path, "y": y_path}


def load_feature_split(split_name: str, feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved feature split."""
    feature_dir = Path(feature_dir)
    x_path = feature_dir / f"X_{split_name}.npy"
    y_path = feature_dir / f"y_{split_name}.npy"
    return load_numpy(x_path), load_numpy(y_path)


def feature_files_exist(feature_dir: str | Path, split_names: tuple[str, ...] = ("train", "val", "test")) -> bool:
    """Check whether cached feature files exist for all requested splits."""
    feature_dir = Path(feature_dir)
    return all(
        (feature_dir / f"X_{split}.npy").exists() and (feature_dir / f"y_{split}.npy").exists()
        for split in split_names
    )


def create_run_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    """Create and return a run directory."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        base_dir = base_dir / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_experiment_result(result: dict, output_dir: str | Path, run_name: str) -> dict[str, Path]:
    """Save structured experiment artifacts and return the created file paths."""
    run_dir = create_run_dir(output_dir, run_name)
    paths: dict[str, Path] = {}

    if "metrics" in result:
        metrics_path = run_dir / "metrics.json"
        save_json(result["metrics"], metrics_path)
        paths["metrics"] = metrics_path

    if "config" in result:
        config_path = run_dir / "config.json"
        save_json(result["config"], config_path)
        paths["config"] = config_path

    dataframe_artifacts = {
        "results_df": "results.csv",
        "report": "report.csv",
        "confusion_matrix": "confusion_matrix.csv",
        "predictions": "predictions.csv",
        "wrong_predictions": "wrong_predictions.csv",
    }

    for key, filename in dataframe_artifacts.items():
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            artifact_path = run_dir / filename
            save_dataframe(value, artifact_path)
            paths[key] = artifact_path

    if "summary" in result:
        summary_value = result["summary"]
        if isinstance(summary_value, pd.DataFrame):
            summary_path = run_dir / "summary.csv"
            save_dataframe(summary_value, summary_path)
        else:
            summary_path = run_dir / "summary.json"
            save_json(summary_value, summary_path)
        paths["summary"] = summary_path

    return paths


def copy_dataframe_images(
    df: pd.DataFrame,
    output_dir: str | Path,
    path_col: str = "path",
    label_col: str = "label_name",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy indexed images into ``output_dir/<label_name>/`` folders.

    This is optional utility code for exporting a cleaned dataset. It is not used
    by the core pipeline, but it is useful for inspection and submission artifacts.
    """

    if path_col not in df.columns:
        raise KeyError(f"path_col not found in dataframe: {path_col}")
    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    output_dir = Path(output_dir)
    copied_rows = []

    for _, row in df.iterrows():
        src = Path(row[path_col])
        label_name = str(row[label_col])
        dst_dir = output_dir / label_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists() and not overwrite:
            stem, suffix = dst.stem, dst.suffix
            counter = 1
            while dst.exists():
                dst = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(src, dst)
        copied = row.to_dict()
        copied["copied_path"] = str(dst)
        copied_rows.append(copied)

    return pd.DataFrame(copied_rows)


__all__ = [
    "save_json",
    "load_json",
    "save_dataframe",
    "load_dataframe",
    "save_numpy",
    "load_numpy",
    "save_pickle",
    "load_pickle",
    "save_feature_split",
    "load_feature_split",
    "feature_files_exist",
    "create_run_dir",
    "save_experiment_result",
    "copy_dataframe_images"
]