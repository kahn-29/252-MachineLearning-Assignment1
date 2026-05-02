import cv2
import numpy as np
import pandas as pd
from PIL import Image


def read_image_cv2(path: str):
    """
    Read image using cv2.
    Return BGR image or None.
    """
    ...


def read_image_pil(path: str) -> Image.Image | None:
    """
    Read image using PIL and convert to RGB.
    Return PIL image or None.
    """
    ...


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    Lower value means blurrier image.
    """
    ...


def compute_entropy(gray: np.ndarray) -> float:
    """
    Compute Shannon entropy of grayscale image.
    Lower value means less visual information.
    """
    ...


def compute_brightness_stats(gray: np.ndarray) -> dict:
    """
    Return brightness_mean and brightness_std.
    """
    ...


def compute_gray_tolerance(rgb: np.ndarray) -> dict:
    """
    Measure how close an image is to grayscale.
    Return gray_diff_mean, gray_diff_p95, near_gray_ratio_10.
    """
    ...


def compute_near_mono_metrics(gray: np.ndarray) -> dict:
    """
    Compute dark_ratio, bright_ratio, near_mono_ratio.
    Used to detect almost-black or almost-white images.
    """
    ...


def compute_aspect_metrics(width: int, height: int) -> dict:
    """
    Compute min_side, max_side, and aspect_extremity.
    """
    ...


def compute_saturation_metrics(bgr: np.ndarray) -> dict:
    """
    Compute mean and standard deviation of HSV saturation channel.
    """
    ...


def compute_chromaticity_metrics(rgb: np.ndarray) -> dict:
    """
    Compute chroma_mean, chroma_std, and chromaticity_spread.
    """
    ...


def compute_center_saliency(gray: np.ndarray) -> dict:
    """
    Estimate saliency using gradient/edge magnitude.
    Return center_saliency_ratio and related saliency statistics.
    """
    ...


def compute_compression_artifact(gray: np.ndarray) -> float:
    """
    Estimate JPEG blockiness / compression artifact score.
    """
    ...


def compute_phash(path: str) -> str | None:
    """
    Compute perceptual hash of image.
    Return hash string or None.
    """
    ...


def inspect_image(path: str, label: int, label_name: str) -> dict:
    """
    Compute all audit metrics for one image.
    Return one record dictionary.
    """
    ...


def audit_dataframe(df: pd.DataFrame, cache_path: str | None = None) -> pd.DataFrame:
    """
    Compute audit metrics for all images in df.
    If cache_path exists, load from cache.
    Otherwise compute and save.
    """
    ...


def describe_audit_metrics(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for all quality metrics.
    """
    ...