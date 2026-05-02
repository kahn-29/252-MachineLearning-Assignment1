from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def read_image_cv2(path: str | Path):
    """Read an image with OpenCV and return BGR array or None."""

    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:
        return None


def read_image_pil(path: str | Path) -> Image.Image | None:
    """Read an image with PIL and return an RGB image or None."""

    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Compute Laplacian variance where lower values indicate more blur."""

    if gray is None or gray.size == 0:
        return float("nan")
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_entropy(gray: np.ndarray) -> float:
    """Compute Shannon entropy from grayscale histogram."""

    if gray is None or gray.size == 0:
        return float("nan")

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0

    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob)))


def compute_brightness_stats(gray: np.ndarray) -> dict:
    """Compute brightness mean and standard deviation."""

    if gray is None or gray.size == 0:
        return {"brightness_mean": float("nan"), "brightness_std": float("nan")}

    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
    }


def compute_gray_tolerance(rgb: np.ndarray) -> dict:
    """Measure channel differences to estimate grayscale likeness."""

    if rgb is None or rgb.size == 0:
        return {
            "gray_diff_mean": float("nan"),
            "gray_diff_p95": float("nan"),
            "near_gray_ratio_10": float("nan"),
        }

    rgb_f = rgb.astype(np.float32)
    channel_delta = rgb_f.max(axis=2) - rgb_f.min(axis=2)

    return {
        "gray_diff_mean": float(channel_delta.mean()),
        "gray_diff_p95": float(np.percentile(channel_delta, 95)),
        "near_gray_ratio_10": float((channel_delta <= 10).mean()),
    }


def compute_near_mono_metrics(gray: np.ndarray) -> dict:
    """Compute near-black/near-white ratios and total near-monochrome ratio."""

    if gray is None or gray.size == 0:
        return {
            "dark_ratio": float("nan"),
            "bright_ratio": float("nan"),
            "near_mono_ratio": float("nan"),
        }

    dark_ratio = (gray < 15).mean()
    bright_ratio = (gray > 240).mean()

    return {
        "dark_ratio": float(dark_ratio),
        "bright_ratio": float(bright_ratio),
        "near_mono_ratio": float(dark_ratio + bright_ratio),
    }


def compute_aspect_metrics(width: int, height: int) -> dict:
    """Compute size and aspect metrics from image width and height."""

    if width <= 0 or height <= 0:
        return {
            "width": float("nan"),
            "height": float("nan"),
            "min_side": float("nan"),
            "max_side": float("nan"),
            "aspect_ratio": float("nan"),
            "aspect_extremity": float("nan"),
        }

    aspect_ratio = width / height

    return {
        "width": int(width),
        "height": int(height),
        "min_side": int(min(width, height)),
        "max_side": int(max(width, height)),
        "aspect_ratio": float(aspect_ratio),
        "aspect_extremity": float(max(aspect_ratio, 1.0 / aspect_ratio)),
    }


def compute_saturation_metrics(bgr: np.ndarray) -> dict:
    """Compute mean and std saturation from HSV representation."""

    if bgr is None or bgr.size == 0:
        return {"mean_sat": float("nan"), "std_sat": float("nan")}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return {"mean_sat": float(sat.mean()), "std_sat": float(sat.std())}


def compute_chromaticity_metrics(rgb: np.ndarray) -> dict:
    """Compute simple color variation metrics from RGB image."""

    if rgb is None or rgb.size == 0:
        return {
            "chroma_mean": float("nan"),
            "chroma_std": float("nan"),
            "chromaticity_spread": float("nan"),
        }

    rgb_f = rgb.astype(np.float32)
    chroma = rgb_f.max(axis=2) - rgb_f.min(axis=2)

    normalized = rgb_f / 255.0
    r = normalized[:, :, 0]
    g = normalized[:, :, 1]
    b = normalized[:, :, 2]

    x_big = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y_big = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z_big = 0.0193 * r + 0.1192 * g + 0.9505 * b

    denom = x_big + y_big + z_big
    valid = denom > 1e-6

    if valid.sum() < 2:
        spread = 0.0
    else:
        x = x_big[valid] / denom[valid]
        y = y_big[valid] / denom[valid]
        spread = float(x.std() + y.std())

    return {
        "chroma_mean": float(chroma.mean()),
        "chroma_std": float(chroma.std()),
        "chromaticity_spread": spread,
    }


def compute_center_saliency(gray: np.ndarray) -> dict:
    """Use Sobel magnitude as a lightweight center-saliency proxy."""

    if gray is None or gray.size == 0:
        return {
            "center_saliency_ratio": float("nan"),
            "saliency_mean": float("nan"),
            "saliency_max": float("nan"),
        }

    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    saliency = np.sqrt(gx * gx + gy * gy)

    h, w = saliency.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4

    total = float(saliency.sum()) + 1e-8
    center = float(saliency[y0:y1, x0:x1].sum())

    return {
        "center_saliency_ratio": center / total,
        "saliency_mean": float(saliency.mean()),
        "saliency_max": float(saliency.max()),
    }


def compute_compression_artifact(gray: np.ndarray) -> float:
    """Approximate block-compression artifacts from 8-pixel boundaries."""

    if gray is None or gray.size == 0:
        return float("nan")

    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    arr = gray.astype(np.float32)
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))

    cols = np.arange(7, w - 1, 8)
    rows = np.arange(7, h - 1, 8)

    if cols.size == 0 or rows.size == 0:
        return 0.0

    boundary_x = gx[:, cols].mean()
    boundary_y = gy[rows, :].mean()
    global_grad = (gx.mean() + gy.mean()) / 2.0 + 1e-8

    return float(((boundary_x + boundary_y) / 2.0) / global_grad)


def compute_phash(path: str | Path) -> str | None:
    """Compute perceptual hash using imagehash.phash."""

    try:
        import imagehash
    except ImportError:
        return None

    img = read_image_pil(path)
    if img is None:
        return None

    try:
        return str(imagehash.phash(img, hash_size=8))
    except Exception:
        return None


def _nan_record(path: str, label: int | None, label_name: str | None) -> dict:
    nan = float("nan")
    return {
        "path": path,
        "label": label,
        "label_name": label_name,
        "is_corrupted": True,
        "width": nan,
        "height": nan,
        "min_side": nan,
        "max_side": nan,
        "aspect_ratio": nan,
        "aspect_extremity": nan,
        "brightness_mean": nan,
        "brightness_std": nan,
        "blur_laplacian": nan,
        "entropy": nan,
        "gray_diff_mean": nan,
        "gray_diff_p95": nan,
        "near_gray_ratio_10": nan,
        "dark_ratio": nan,
        "bright_ratio": nan,
        "near_mono_ratio": nan,
        "mean_sat": nan,
        "std_sat": nan,
        "chroma_mean": nan,
        "chroma_std": nan,
        "chromaticity_spread": nan,
        "center_saliency_ratio": nan,
        "saliency_mean": nan,
        "saliency_max": nan,
        "compression_artifact": nan,
        "phash": None,
    }


def inspect_image(
    path: str | Path,
    label: int | None = None,
    label_name: str | None = None,
    compute_hash: bool = True,
) -> dict:
    """Inspect a single image and return a dictionary of audit metrics."""

    path_str = str(path)
    bgr = read_image_cv2(path_str)

    if bgr is None:
        return _nan_record(path_str, label, label_name)

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    record = {
        "path": path_str,
        "label": label,
        "label_name": label_name,
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
    record["phash"] = compute_phash(path_str) if compute_hash else None

    return record


def audit_dataframe(
    df: pd.DataFrame,
    path_col: str = "path",
    label_col: str = "label",
    label_name_col: str = "label_name",
    compute_hash: bool = True,
) -> pd.DataFrame:
    """Run inspect_image for each row and return an audit dataframe."""

    if path_col not in df.columns:
        raise KeyError(f"Required column not found: {path_col}")

    records: list[dict] = []

    for _, row in df.iterrows():
        path = row[path_col]
        label = row[label_col] if label_col in df.columns else None
        label_name = row[label_name_col] if label_name_col in df.columns else None
        records.append(
            inspect_image(
                path=path,
                label=label,
                label_name=label_name,
                compute_hash=compute_hash,
            )
        )

    return pd.DataFrame(records).reset_index(drop=True)


def describe_audit_metrics(
    audit_df: pd.DataFrame,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return descriptive statistics for selected audit metrics."""

    if metrics is None:
        metrics = [
            "blur_laplacian",
            "entropy",
            "brightness_mean",
            "brightness_std",
            "gray_diff_mean",
            "near_gray_ratio_10",
            "near_mono_ratio",
            "mean_sat",
            "chroma_mean",
            "center_saliency_ratio",
            "compression_artifact",
            "min_side",
            "aspect_extremity",
        ]

    metric_cols = [m for m in metrics if m in audit_df.columns]
    if not metric_cols:
        raise ValueError("No valid metric columns found in audit_df.")

    return audit_df[metric_cols].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    ).T
