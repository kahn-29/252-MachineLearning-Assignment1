import pandas as pd
import numpy as np


def hamming_distance_hash(hash_1: str, hash_2: str) -> int:
    """
    Compute Hamming distance between two perceptual hashes.
    """
    ...


def mark_near_duplicates(
    audit_df: pd.DataFrame,
    hamming_threshold: int = 4,
    hash_col: str = "phash"
) -> pd.DataFrame:
    """
    Detect near-duplicate images using perceptual hash.
    Add columns:
    duplicate_cluster, is_duplicate.
    Keep the highest-quality image in each cluster.
    """
    ...


def compute_quality_score(audit_df: pd.DataFrame) -> pd.Series:
    """
    Compute quality score for choosing the best image in a duplicate cluster.
    Higher score means better image.
    """
    ...


def build_cleaning_mask(
    audit_df: pd.DataFrame,
    cleaning_config: dict
) -> pd.Series:
    """
    Apply hard filters and soft-flag logic.
    Return boolean mask where True means image is kept.
    """
    ...


def compute_soft_flags(
    audit_df: pd.DataFrame,
    cleaning_config: dict
) -> pd.Series:
    """
    Count how many soft-quality warnings each image has.
    Soft flags may include low entropy, low saturation, low saliency,
    high compression artifact, low brightness variance, etc.
    """
    ...


def assign_removal_reasons(
    audit_df: pd.DataFrame,
    cleaning_config: dict
) -> pd.DataFrame:
    """
    Add removal_reason column explaining why each removed image was removed.
    """
    ...


def apply_cleaning(
    audit_df: pd.DataFrame,
    cleaning_config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return clean_df and removed_df.
    clean_df contains kept images.
    removed_df contains removed images with reasons.
    """
    ...


def summarize_cleaning(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    removed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize before/after cleaning by class.
    """
    ...


def evaluate_cleaning_retention(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame
) -> dict:
    """
    Compute retention rate, removal rate, and class balance shift.
    """
    ...