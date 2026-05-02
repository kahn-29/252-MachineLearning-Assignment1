# modules/image_audit.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def read_image_cv2(path: str | Path):
    """
    Read image using OpenCV.

    Parameters
    ----------
    path : str | Path
        Image path.

    Returns
    -------
    np.ndarray | None
        BGR image if readable, otherwise None.
    """

    path = str(path)
    img = cv2.imread(path)

    return img


def read_image_pil(path: str | Path) -> Image.Image | None:
    """
    Read image using PIL and convert it to RGB.

    Parameters
    ----------
    path : str | Path
        Image path.

    Returns
    -------
    PIL.Image.Image | None
        RGB image if readable, otherwise None.
    """

    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.

    A lower value usually indicates a blurrier image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    float
        Laplacian variance.
    """

    if gray is None or gray.size == 0:
        return np.nan

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_entropy(gray: np.ndarray) -> float:
    """
    Compute Shannon entropy of a grayscale image.

    Lower entropy indicates less visual information.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    float
        Image entropy.
    """

    if gray is None or gray.size == 0:
        return np.nan

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist > 0]

    if hist.sum() == 0:
        return 0.0

    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))

    return float(entropy)


def compute_brightness_stats(gray: np.ndarray) -> dict:
    """
    Compute brightness mean and standard deviation.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    dict
        brightness_mean, brightness_std.
    """

    if gray is None or gray.size == 0:
        return {
            "brightness_mean": np.nan,
            "brightness_std": np.nan,
        }

    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
    }


def compute_gray_tolerance(rgb: np.ndarray) -> dict:
    """
    Measure how close an RGB image is to grayscale.

    The metric is based on the difference among R, G, and B channels.
    A low channel difference means the image is closer to grayscale.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image.

    Returns
    -------
    dict
        gray_diff_mean, gray_diff_p95, near_gray_ratio_10.
    """

    if rgb is None or rgb.size == 0:
        return {
            "gray_diff_mean": np.nan,
            "gray_diff_p95": np.nan,
            "near_gray_ratio_10": np.nan,
        }

    arr = rgb.astype(np.float32)
    channel_range = arr.max(axis=2) - arr.min(axis=2)

    return {
        "gray_diff_mean": float(channel_range.mean()),
        "gray_diff_p95": float(np.percentile(channel_range, 95)),
        "near_gray_ratio_10": float((channel_range <= 10).mean()),
    }


def compute_near_mono_metrics(gray: np.ndarray) -> dict:
    """
    Compute near-monochrome indicators.

    dark_ratio:
        Ratio of pixels with intensity lower than 15.
    bright_ratio:
        Ratio of pixels with intensity higher than 240.
    near_mono_ratio:
        Ratio of pixels that are either very dark or very bright.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    dict
        dark_ratio, bright_ratio, near_mono_ratio.
    """

    if gray is None or gray.size == 0:
        return {
            "dark_ratio": np.nan,
            "bright_ratio": np.nan,
            "near_mono_ratio": np.nan,
        }

    dark_ratio = (gray < 15).mean()
    bright_ratio = (gray > 240).mean()
    near_mono_ratio = ((gray < 15) | (gray > 240)).mean()

    return {
        "dark_ratio": float(dark_ratio),
        "bright_ratio": float(bright_ratio),
        "near_mono_ratio": float(near_mono_ratio),
    }


def compute_aspect_metrics(width: int, height: int) -> dict:
    """
    Compute image size and aspect-ratio related metrics.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.

    Returns
    -------
    dict
        width, height, min_side, max_side, aspect_ratio, aspect_extremity.
    """

    if width <= 0 or height <= 0:
        return {
            "width": np.nan,
            "height": np.nan,
            "min_side": np.nan,
            "max_side": np.nan,
            "aspect_ratio": np.nan,
            "aspect_extremity": np.nan,
        }

    aspect_ratio = width / height
    aspect_extremity = max(width / height, height / width)

    return {
        "width": int(width),
        "height": int(height),
        "min_side": int(min(width, height)),
        "max_side": int(max(width, height)),
        "aspect_ratio": float(aspect_ratio),
        "aspect_extremity": float(aspect_extremity),
    }


def compute_saturation_metrics(bgr: np.ndarray) -> dict:
    """
    Compute saturation statistics from HSV colour space.

    Parameters
    ----------
    bgr : np.ndarray
        BGR image.

    Returns
    -------
    dict
        mean_sat, std_sat.
    """

    if bgr is None or bgr.size == 0:
        return {
            "mean_sat": np.nan,
            "std_sat": np.nan,
        }

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    return {
        "mean_sat": float(saturation.mean()),
        "std_sat": float(saturation.std()),
    }


def compute_chromaticity_metrics(rgb: np.ndarray) -> dict:
    """
    Compute simple chromaticity and colour variation metrics.

    chroma_mean:
        Mean difference between the maximum and minimum RGB channels.
    chromaticity_spread:
        Spread of xy chromaticity coordinates in approximate XYZ space.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image.

    Returns
    -------
    dict
        chroma_mean, chroma_std, chromaticity_spread.
    """

    if rgb is None or rgb.size == 0:
        return {
            "chroma_mean": np.nan,
            "chroma_std": np.nan,
            "chromaticity_spread": np.nan,
        }

    arr = rgb.astype(np.float32)
    chroma = arr.max(axis=2) - arr.min(axis=2)

    normalized = arr / 255.0
    r = normalized[:, :, 0]
    g = normalized[:, :, 1]
    b = normalized[:, :, 2]

    x_big = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y_big = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z_big = 0.0193 * r + 0.1192 * g + 0.9505 * b

    denom = x_big + y_big + z_big
    mask = denom > 1e-6

    if mask.sum() < 100:
        chromaticity_spread = 0.0
    else:
        x = x_big[mask] / denom[mask]
        y = y_big[mask] / denom[mask]
        chromaticity_spread = float(x.std() + y.std())

    return {
        "chroma_mean": float(chroma.mean()),
        "chroma_std": float(chroma.std()),
        "chromaticity_spread": chromaticity_spread,
    }


def compute_center_saliency(gray: np.ndarray) -> dict:
    """
    Estimate object saliency and centering using gradient magnitude.

    This does not require opencv-contrib. It uses Sobel edges as a lightweight
    proxy for visual saliency.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    dict
        center_saliency_ratio, saliency_mean, saliency_max.
    """

    if gray is None or gray.size == 0:
        return {
            "center_saliency_ratio": np.nan,
            "saliency_mean": np.nan,
            "saliency_max": np.nan,
        }

    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)

    saliency = np.sqrt(gx * gx + gy * gy)
    total_saliency = float(saliency.sum()) + 1e-8

    h, w = saliency.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4

    center_saliency = float(saliency[y0:y1, x0:x1].sum())

    return {
        "center_saliency_ratio": center_saliency / total_saliency,
        "saliency_mean": float(saliency.mean()),
        "saliency_max": float(saliency.max()),
    }


def compute_compression_artifact(gray: np.ndarray) -> float:
    """
    Estimate JPEG blockiness / compression artifact score.

    The score compares gradient changes along 8-pixel block boundaries
    against the average gradient magnitude.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    float
        Compression artifact score.
    """

    if gray is None or gray.size == 0:
        return np.nan

    h, w = gray.shape

    if h < 16 or w < 16:
        return 0.0

    g = gray.astype(np.float32)

    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))

    block_cols = np.arange(7, w - 1, 8)
    block_rows = np.arange(7, h - 1, 8)

    if len(block_cols) == 0 or len(block_rows) == 0:
        return 0.0

    boundary_x = gx[:, block_cols].mean()
    boundary_y = gy[block_rows, :].mean()

    global_gradient = (gx.mean() + gy.mean()) / 2.0 + 1e-8
    artifact_score = ((boundary_x + boundary_y) / 2.0) / global_gradient

    return float(artifact_score)


def compute_phash(path: str | Path) -> str | None:
    """
    Compute perceptual hash of an image.

    This requires the imagehash package.

    Parameters
    ----------
    path : str | Path
        Image path.

    Returns
    -------
    str | None
        Perceptual hash string if successful, otherwise None.
    """

    try:
        import imagehash
    except ImportError as exc:
        raise ImportError(
            "imagehash is required for perceptual hashing. "
            "Install it with: pip install imagehash"
        ) from exc

    try:
        img = Image.open(path).convert("RGB")
        return str(imagehash.phash(img, hash_size=8))
    except Exception:
        return None


def inspect_image(
    path: str | Path,
    label: int | None = None,
    label_name: str | None = None,
    compute_hash: bool = True
) -> dict:
    """
    Compute audit metrics for a single image.

    Parameters
    ----------
    path : str | Path
        Image path.
    label : int | None
        Numeric label.
    label_name : str | None
        Human-readable label.
    compute_hash : bool
        Whether to compute perceptual hash.

    Returns
    -------
    dict
        One image audit record.
    """

    path = str(path)

    base_record = {
        "path": path,
        "label": label,
        "label_name": label_name,
    }

    bgr = read_image_cv2(path)

    if bgr is None:
        return {
            **base_record,
            "is_corrupted": True,
            "width": np.nan,
            "height": np.nan,
            "min_side": np.nan,
            "max_side": np.nan,
            "aspect_ratio": np.nan,
            "aspect_extremity": np.nan,
            "brightness_mean": np.nan,
            "brightness_std": np.nan,
            "blur_laplacian": np.nan,
            "entropy": np.nan,
            "gray_diff_mean": np.nan,
            "gray_diff_p95": np.nan,
            "near_gray_ratio_10": np.nan,
            "dark_ratio": np.nan,
            "bright_ratio": np.nan,
            "near_mono_ratio": np.nan,
            "mean_sat": np.nan,
            "std_sat": np.nan,
            "chroma_mean": np.nan,
            "chroma_std": np.nan,
            "chromaticity_spread": np.nan,
            "center_saliency_ratio": np.nan,
            "saliency_mean": np.nan,
            "saliency_max": np.nan,
            "compression_artifact": np.nan,
            "phash": None,
        }

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    record = {
        **base_record,
        "is_corrupted": False,
    }

    record.update(compute_aspect_metrics(width=w, height=h))
    record.update(compute_brightness_stats(gray))
    record["blur_laplacian"] = compute_laplacian_variance(gray)
    record["entropy"] = compute_entropy(gray)
    record.update(compute_gray_tolerance(rgb))
    record.update(compute_near_mono_metrics(gray))
    record.update(compute_saturation_metrics(bgr))
    record.update(compute_chromaticity_metrics(rgb))
    record.update(compute_center_saliency(gray))
    record["compression_artifact"] = compute_compression_artifact(gray)

    if compute_hash:
        record["phash"] = compute_phash(path)
    else:
        record["phash"] = None

    return record


def audit_dataframe(
    df: pd.DataFrame,
    cache_path: str | Path | None = None,
    path_col: str = "path",
    label_col: str = "label",
    label_name_col: str = "label_name",
    compute_hash: bool = True,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Compute audit metrics for all images in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Expected columns: path, label, label_name.
    cache_path : str | Path | None
        Optional CSV cache path.
    path_col : str
        Column containing image paths.
    label_col : str
        Column containing numeric labels.
    label_name_col : str
        Column containing label names.
    compute_hash : bool
        Whether to compute perceptual hash.
    force_recompute : bool
        If True, ignore cache and recompute all metrics.

    Returns
    -------
    pd.DataFrame
        Audit dataframe.
    """

    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_recompute:
            return pd.read_csv(cache_path)

    required_cols = [path_col]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column not found: {col}")

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Auditing images"):
        path = row[path_col]
        label = row[label_col] if label_col in df.columns else None
        label_name = row[label_name_col] if label_name_col in df.columns else None

        try:
            record = inspect_image(
                path=path,
                label=label,
                label_name=label_name,
                compute_hash=compute_hash,
            )
        except Exception:
            record = {
                "path": path,
                "label": label,
                "label_name": label_name,
                "is_corrupted": True,
            }

        records.append(record)

    audit_df = pd.DataFrame(records).reset_index(drop=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(cache_path, index=False)

    return audit_df


def get_quality_metric_columns() -> list[str]:
    """
    Return the standard list of audit metric columns.

    Returns
    -------
    list[str]
        Metric column names.
    """

    return [
        "width",
        "height",
        "min_side",
        "max_side",
        "aspect_ratio",
        "aspect_extremity",
        "brightness_mean",
        "brightness_std",
        "blur_laplacian",
        "entropy",
        "gray_diff_mean",
        "gray_diff_p95",
        "near_gray_ratio_10",
        "dark_ratio",
        "bright_ratio",
        "near_mono_ratio",
        "mean_sat",
        "std_sat",
        "chroma_mean",
        "chroma_std",
        "chromaticity_spread",
        "center_saliency_ratio",
        "saliency_mean",
        "saliency_max",
        "compression_artifact",
    ]


def describe_audit_metrics(
    audit_df: pd.DataFrame,
    metrics: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Return descriptive statistics for image audit metrics.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    metrics : Iterable[str] | None
        Metrics to describe. If None, standard metric columns are used.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics.
    """

    if metrics is None:
        metrics = get_quality_metric_columns()

    metrics = [m for m in metrics if m in audit_df.columns]

    if not metrics:
        raise ValueError("No valid metric columns found in audit dataframe.")

    return audit_df[metrics].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    ).T


def add_basic_quality_flags(
    audit_df: pd.DataFrame,
    blur_threshold: float = 50.0,
    min_side_threshold: int = 64,
    near_mono_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Add simple quality flags for quick inspection.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    blur_threshold : float
        Minimum acceptable Laplacian variance.
    min_side_threshold : int
        Minimum acceptable image side.
    near_mono_threshold : float
        Maximum acceptable near-monochrome ratio.

    Returns
    -------
    pd.DataFrame
        Audit dataframe with additional flag columns.
    """

    output = audit_df.copy()

    output["flag_blurry"] = output["blur_laplacian"] < blur_threshold
    output["flag_undersized"] = output["min_side"] < min_side_threshold
    output["flag_near_mono"] = output["near_mono_ratio"] > near_mono_threshold
    output["flag_corrupted"] = output["is_corrupted"].fillna(False)

    output["n_basic_flags"] = (
        output["flag_blurry"].astype(int)
        + output["flag_undersized"].astype(int)
        + output["flag_near_mono"].astype(int)
        + output["flag_corrupted"].astype(int)
    )

    return output


def sample_extreme_images(
    audit_df: pd.DataFrame,
    metric: str,
    n: int = 12,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Return images with extreme values for a selected metric.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    metric : str
        Metric column.
    n : int
        Number of samples to return.
    ascending : bool
        If True, return lowest values. Otherwise return highest values.

    Returns
    -------
    pd.DataFrame
        Sampled extreme images.
    """

    if metric not in audit_df.columns:
        raise KeyError(f"Metric not found in audit dataframe: {metric}")

    return (
        audit_df
        .dropna(subset=[metric])
        .sort_values(metric, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )


def summarize_corruption(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize corrupted and valid images by class.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """

    if "is_corrupted" not in audit_df.columns:
        raise KeyError("Column 'is_corrupted' not found in audit dataframe.")

    summary = (
        audit_df
        .groupby(["label_name", "is_corrupted"])
        .size()
        .reset_index(name="count")
    )

    total_by_class = summary.groupby("label_name")["count"].transform("sum")
    summary["percentage"] = summary["count"] / total_by_class * 100

    return summary