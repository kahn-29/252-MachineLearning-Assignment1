# modules/cleaning.py

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class UnionFind:
    """
    Lightweight Union-Find structure for duplicate clustering.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def hamming_distance_hash(hash_1: str | None, hash_2: str | None) -> int:
    """
    Compute Hamming distance between two hexadecimal perceptual hashes.

    Parameters
    ----------
    hash_1 : str | None
        First perceptual hash.
    hash_2 : str | None
        Second perceptual hash.

    Returns
    -------
    int
        Hamming distance. Returns a large number if either hash is invalid.
    """

    if hash_1 is None or hash_2 is None:
        return 10**9

    try:
        return (int(str(hash_1), 16) ^ int(str(hash_2), 16)).bit_count()
    except Exception:
        return 10**9


def compute_quality_score(audit_df: pd.DataFrame) -> pd.Series:
    """
    Compute a relative image quality score.

    The score is mainly used to select which image to keep inside
    a near-duplicate cluster. Higher score indicates better image quality.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.

    Returns
    -------
    pd.Series
        Quality score for each image.
    """

    df = audit_df.copy()

    def rank_col(col: str, ascending: bool = True) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)

        values = df[col].replace([np.inf, -np.inf], np.nan)

        if values.notna().sum() == 0:
            return pd.Series(0.0, index=df.index)

        return values.rank(pct=True, ascending=ascending).fillna(0.0)

    score = (
        rank_col("blur_laplacian", ascending=True)
        + rank_col("entropy", ascending=True)
        + rank_col("min_side", ascending=True)
        + rank_col("brightness_std", ascending=True)
        + rank_col("center_saliency_ratio", ascending=True)
        - rank_col("compression_artifact", ascending=True)
    )

    if "is_corrupted" in df.columns:
        score = score - df["is_corrupted"].fillna(False).astype(int) * 10

    return score


def mark_near_duplicates(
    audit_df: pd.DataFrame,
    hamming_threshold: int = 4,
    hash_col: str = "phash",
    bands: int = 8,
) -> pd.DataFrame:
    """
    Detect near-duplicate images using perceptual hash.

    The function uses a simple locality-sensitive hashing strategy to reduce
    the number of pairwise comparisons. Candidate pairs are verified using
    Hamming distance.

    Within each duplicate cluster, only the image with the highest quality
    score is kept. Other images are marked as duplicates.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe containing perceptual hash column.
    hamming_threshold : int
        Maximum Hamming distance for two images to be considered near-duplicates.
    hash_col : str
        Column containing perceptual hashes.
    bands : int
        Number of hash bands used for candidate generation.

    Returns
    -------
    pd.DataFrame
        Audit dataframe with duplicate_cluster, duplicate_group_size,
        is_duplicate, and is_duplicate_representative columns.
    """

    if hash_col not in audit_df.columns:
        raise KeyError(f"Hash column not found: {hash_col}")

    df = audit_df.copy().reset_index(drop=True)
    n = len(df)

    hashes = df[hash_col].astype("object").where(df[hash_col].notna(), None).tolist()
    valid_mask = np.array([h is not None and isinstance(h, str) for h in hashes])

    uf = UnionFind(n)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    if bands <= 0:
        raise ValueError("bands must be a positive integer.")

    bits_per_band = max(1, 64 // bands)

    for i, h in enumerate(hashes):
        if not valid_mask[i]:
            continue

        try:
            value = int(h, 16)
        except Exception:
            continue

        for band_idx in range(bands):
            shift = band_idx * bits_per_band
            band_value = (value >> shift) & ((1 << bits_per_band) - 1)
            buckets[(band_idx, band_value)].append(i)

    candidate_pairs = set()

    for bucket in buckets.values():
        if len(bucket) < 2:
            continue

        for a, b in combinations(bucket, 2):
            if a < b:
                candidate_pairs.add((a, b))
            else:
                candidate_pairs.add((b, a))

    for a, b in candidate_pairs:
        if hamming_distance_hash(hashes[a], hashes[b]) <= hamming_threshold:
            uf.union(a, b)

    clusters = np.array([uf.find(i) for i in range(n)])

    quality_score = compute_quality_score(df)

    representative = np.zeros(n, dtype=bool)

    temp = pd.DataFrame(
        {
            "idx": np.arange(n),
            "cluster": clusters,
            "quality_score": quality_score.values,
            "has_valid_hash": valid_mask,
        }
    )

    for _, group in temp.groupby("cluster"):
        if len(group) == 1:
            representative[group["idx"].iloc[0]] = True
            continue

        valid_group = group[group["has_valid_hash"]]

        if len(valid_group) == 0:
            representative[group["idx"].iloc[0]] = True
            continue

        best_idx = (
            valid_group
            .sort_values("quality_score", ascending=False)
            ["idx"]
            .iloc[0]
        )
        representative[best_idx] = True

    group_sizes = pd.Series(clusters).map(pd.Series(clusters).value_counts()).values

    df["duplicate_cluster"] = clusters
    df["duplicate_group_size"] = group_sizes
    df["is_duplicate_representative"] = representative
    df["is_duplicate"] = (group_sizes > 1) & (~representative)

    return df


def compute_soft_flags(
    audit_df: pd.DataFrame,
    cleaning_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute soft-quality flags.

    Soft flags are warnings, not automatic removal rules by themselves.
    An image is removed by soft flags only when the number of triggered
    soft flags exceeds the configured limit.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    cleaning_config : dict
        Cleaning configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one boolean column per soft flag and
        soft_flag_count.
    """

    df = audit_df.copy()
    flags = pd.DataFrame(index=df.index)

    def has_col(col: str) -> bool:
        return col in df.columns

    if cleaning_config.get("gray_diff_mean_min") is not None and has_col("gray_diff_mean"):
        flags["soft_low_gray_difference"] = (
            df["gray_diff_mean"] < cleaning_config["gray_diff_mean_min"]
        )
    else:
        flags["soft_low_gray_difference"] = False

    if cleaning_config.get("entropy_min") is not None and has_col("entropy"):
        flags["soft_low_entropy"] = (
            df["entropy"] < cleaning_config["entropy_min"]
        )
    else:
        flags["soft_low_entropy"] = False

    if cleaning_config.get("entropy_max") is not None and has_col("entropy"):
        flags["soft_high_entropy"] = (
            df["entropy"] > cleaning_config["entropy_max"]
        )
    else:
        flags["soft_high_entropy"] = False

    if cleaning_config.get("mean_sat_min") is not None and has_col("mean_sat"):
        flags["soft_low_saturation"] = (
            df["mean_sat"] < cleaning_config["mean_sat_min"]
        )
    else:
        flags["soft_low_saturation"] = False

    if cleaning_config.get("mean_sat_max") is not None and has_col("mean_sat"):
        flags["soft_high_saturation"] = (
            df["mean_sat"] > cleaning_config["mean_sat_max"]
        )
    else:
        flags["soft_high_saturation"] = False

    if cleaning_config.get("chroma_mean_min") is not None and has_col("chroma_mean"):
        flags["soft_low_chroma"] = (
            df["chroma_mean"] < cleaning_config["chroma_mean_min"]
        )
    else:
        flags["soft_low_chroma"] = False

    if (
        cleaning_config.get("center_saliency_ratio_min") is not None
        and has_col("center_saliency_ratio")
    ):
        flags["soft_low_center_saliency"] = (
            df["center_saliency_ratio"]
            < cleaning_config["center_saliency_ratio_min"]
        )
    else:
        flags["soft_low_center_saliency"] = False

    if (
        cleaning_config.get("compression_artifact_max") is not None
        and has_col("compression_artifact")
    ):
        flags["soft_high_compression_artifact"] = (
            df["compression_artifact"]
            > cleaning_config["compression_artifact_max"]
        )
    else:
        flags["soft_high_compression_artifact"] = False

    if cleaning_config.get("brightness_std_min") is not None and has_col("brightness_std"):
        flags["soft_low_brightness_variance"] = (
            df["brightness_std"] < cleaning_config["brightness_std_min"]
        )
    else:
        flags["soft_low_brightness_variance"] = False

    if cleaning_config.get("brightness_std_max") is not None and has_col("brightness_std"):
        flags["soft_high_brightness_variance"] = (
            df["brightness_std"] > cleaning_config["brightness_std_max"]
        )
    else:
        flags["soft_high_brightness_variance"] = False

    flags = flags.fillna(False).astype(bool)
    flags["soft_flag_count"] = flags.sum(axis=1)

    return flags


def build_cleaning_mask(
    audit_df: pd.DataFrame,
    cleaning_config: dict[str, Any],
) -> pd.Series:
    """
    Build a boolean cleaning mask.

    True means the image is kept.
    False means the image is removed.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    cleaning_config : dict
        Cleaning configuration.

    Returns
    -------
    pd.Series
        Boolean keep mask.
    """

    df = audit_df.copy()
    mask = pd.Series(True, index=df.index)

    remove_corrupted = cleaning_config.get("remove_corrupted", True)
    remove_duplicates = cleaning_config.get("remove_duplicates", True)

    if remove_corrupted and "is_corrupted" in df.columns:
        mask &= ~df["is_corrupted"].fillna(False).astype(bool)

    if remove_duplicates and "is_duplicate" in df.columns:
        mask &= ~df["is_duplicate"].fillna(False).astype(bool)

    if cleaning_config.get("blur_laplacian_min") is not None:
        mask &= df["blur_laplacian"] >= cleaning_config["blur_laplacian_min"]

    if cleaning_config.get("min_side_min") is not None:
        mask &= df["min_side"] >= cleaning_config["min_side_min"]

    if cleaning_config.get("aspect_extremity_max") is not None:
        mask &= df["aspect_extremity"] <= cleaning_config["aspect_extremity_max"]

    if cleaning_config.get("near_mono_ratio_max") is not None:
        mask &= df["near_mono_ratio"] <= cleaning_config["near_mono_ratio_max"]

    entropy_near_mono_max = cleaning_config.get("entropy_near_mono_max", None)

    if cleaning_config.get("dark_ratio_max") is not None:
        dark_bad = df["dark_ratio"] > cleaning_config["dark_ratio_max"]

        if entropy_near_mono_max is not None and "entropy" in df.columns:
            dark_bad &= df["entropy"] < entropy_near_mono_max

        mask &= ~dark_bad.fillna(False)

    if cleaning_config.get("bright_ratio_max") is not None:
        bright_bad = df["bright_ratio"] > cleaning_config["bright_ratio_max"]

        if entropy_near_mono_max is not None and "entropy" in df.columns:
            bright_bad &= df["entropy"] < entropy_near_mono_max

        mask &= ~bright_bad.fillna(False)

    soft_flags = compute_soft_flags(df, cleaning_config)
    max_soft_flags = cleaning_config.get("max_soft_flags", None)

    if max_soft_flags is not None:
        mask &= soft_flags["soft_flag_count"] <= max_soft_flags

    return mask.fillna(False).astype(bool)


def get_removal_reasons_for_row(
    row: pd.Series,
    cleaning_config: dict[str, Any],
    soft_flag_count: int | None = None,
) -> list[str]:
    """
    Return removal reasons for a single image.

    Parameters
    ----------
    row : pd.Series
        One row from audit dataframe.
    cleaning_config : dict
        Cleaning configuration.
    soft_flag_count : int | None
        Number of soft flags triggered.

    Returns
    -------
    list[str]
        Removal reasons.
    """

    reasons = []

    if cleaning_config.get("remove_corrupted", True):
        if bool(row.get("is_corrupted", False)):
            reasons.append("corrupted")

    if cleaning_config.get("remove_duplicates", True):
        if bool(row.get("is_duplicate", False)):
            reasons.append("near_duplicate")

    if cleaning_config.get("blur_laplacian_min") is not None:
        if row.get("blur_laplacian", np.inf) < cleaning_config["blur_laplacian_min"]:
            reasons.append("blurry")

    if cleaning_config.get("min_side_min") is not None:
        if row.get("min_side", np.inf) < cleaning_config["min_side_min"]:
            reasons.append("undersized")

    if cleaning_config.get("aspect_extremity_max") is not None:
        if row.get("aspect_extremity", 0) > cleaning_config["aspect_extremity_max"]:
            reasons.append("extreme_aspect_ratio")

    if cleaning_config.get("near_mono_ratio_max") is not None:
        if row.get("near_mono_ratio", 0) > cleaning_config["near_mono_ratio_max"]:
            reasons.append("near_monochrome")

    entropy_near_mono_max = cleaning_config.get("entropy_near_mono_max", None)

    if cleaning_config.get("dark_ratio_max") is not None:
        dark_bad = row.get("dark_ratio", 0) > cleaning_config["dark_ratio_max"]

        if entropy_near_mono_max is not None:
            dark_bad = dark_bad and row.get("entropy", np.inf) < entropy_near_mono_max

        if dark_bad:
            reasons.append("near_mono_dark")

    if cleaning_config.get("bright_ratio_max") is not None:
        bright_bad = row.get("bright_ratio", 0) > cleaning_config["bright_ratio_max"]

        if entropy_near_mono_max is not None:
            bright_bad = bright_bad and row.get("entropy", np.inf) < entropy_near_mono_max

        if bright_bad:
            reasons.append("near_mono_bright")

    max_soft_flags = cleaning_config.get("max_soft_flags", None)

    if max_soft_flags is not None and soft_flag_count is not None:
        if soft_flag_count > max_soft_flags:
            reasons.append(f"too_many_soft_flags_{soft_flag_count}")

    return reasons


def assign_removal_reasons(
    audit_df: pd.DataFrame,
    cleaning_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Add cleaning decision and removal reason columns.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    cleaning_config : dict
        Cleaning configuration.

    Returns
    -------
    pd.DataFrame
        Dataframe with is_kept, soft_flag_count, and removal_reason columns.
    """

    df = audit_df.copy()
    mask = build_cleaning_mask(df, cleaning_config)
    soft_flags = compute_soft_flags(df, cleaning_config)

    df["is_kept"] = mask.values
    df["soft_flag_count"] = soft_flags["soft_flag_count"].values

    reasons = []

    for idx, row in df.iterrows():
        if mask.loc[idx]:
            reasons.append("kept")
            continue

        row_reasons = get_removal_reasons_for_row(
            row=row,
            cleaning_config=cleaning_config,
            soft_flag_count=int(soft_flags.loc[idx, "soft_flag_count"]),
        )

        if len(row_reasons) == 0:
            row_reasons = ["removed_by_combined_rule"]

        reasons.append(", ".join(row_reasons))

    df["removal_reason"] = reasons

    soft_flag_cols = [c for c in soft_flags.columns if c != "soft_flag_count"]

    for col in soft_flag_cols:
        df[col] = soft_flags[col].values

    return df


def apply_cleaning(
    audit_df: pd.DataFrame,
    cleaning_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply cleaning rules and return clean and removed dataframes.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    cleaning_config : dict
        Cleaning configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        clean_df, removed_df.
    """

    decision_df = assign_removal_reasons(audit_df, cleaning_config)

    clean_df = decision_df[decision_df["is_kept"]].copy().reset_index(drop=True)
    removed_df = decision_df[~decision_df["is_kept"]].copy().reset_index(drop=True)

    return clean_df, removed_df


def summarize_cleaning(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    label_col: str = "label_name",
) -> pd.DataFrame:
    """
    Summarize dataset size and class distribution before and after cleaning.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original dataframe.
    clean_df : pd.DataFrame
        Cleaned dataframe.
    removed_df : pd.DataFrame
        Removed dataframe.
    label_col : str
        Class-name column.

    Returns
    -------
    pd.DataFrame
        Cleaning summary table.
    """

    records = []

    for group_name, df in [
        ("before_cleaning", raw_df),
        ("after_cleaning", clean_df),
        ("removed", removed_df),
    ]:
        total = len(df)
        counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}

        cat_count = int(counts.get("Cat", 0))
        dog_count = int(counts.get("Dog", 0))

        records.append(
            {
                "group": group_name,
                "total": total,
                "cat": cat_count,
                "dog": dog_count,
                "cat_pct": 100 * cat_count / total if total > 0 else 0,
                "dog_pct": 100 * dog_count / total if total > 0 else 0,
            }
        )

    summary = pd.DataFrame(records)

    before_total = max(summary.loc[summary["group"] == "before_cleaning", "total"].iloc[0], 1)
    summary["pct_of_original"] = summary["total"] / before_total * 100

    return summary


def summarize_removal_reasons(
    removed_df: pd.DataFrame,
    reason_col: str = "removal_reason",
) -> pd.DataFrame:
    """
    Summarize removal reasons.

    If an image has multiple comma-separated reasons, each reason is counted.

    Parameters
    ----------
    removed_df : pd.DataFrame
        Removed dataframe.
    reason_col : str
        Column containing removal reasons.

    Returns
    -------
    pd.DataFrame
        Removal reason summary.
    """

    if len(removed_df) == 0:
        return pd.DataFrame(columns=["reason", "count", "percentage"])

    if reason_col not in removed_df.columns:
        raise KeyError(f"reason_col not found: {reason_col}")

    reason_counts = defaultdict(int)

    for reasons in removed_df[reason_col].fillna("unknown"):
        for reason in str(reasons).split(","):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1

    summary = pd.DataFrame(
        [
            {"reason": reason, "count": count}
            for reason, count in reason_counts.items()
        ]
    )

    summary = summary.sort_values("count", ascending=False).reset_index(drop=True)
    summary["percentage"] = summary["count"] / max(len(removed_df), 1) * 100

    return summary


def evaluate_cleaning_retention(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    label_col: str = "label",
) -> dict[str, float]:
    """
    Compute retention, removal, and class-balance shift after cleaning.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original dataframe.
    clean_df : pd.DataFrame
        Cleaned dataframe.
    label_col : str
        Numeric label column.

    Returns
    -------
    dict
        Retention statistics.
    """

    n_raw = len(raw_df)
    n_clean = len(clean_df)

    if n_raw == 0:
        raise ValueError("raw_df is empty.")

    raw_positive_ratio = raw_df[label_col].mean() if label_col in raw_df.columns else np.nan
    clean_positive_ratio = clean_df[label_col].mean() if label_col in clean_df.columns and n_clean > 0 else np.nan

    return {
        "n_raw": int(n_raw),
        "n_clean": int(n_clean),
        "n_removed": int(n_raw - n_clean),
        "retention_rate": float(n_clean / n_raw),
        "removal_rate": float((n_raw - n_clean) / n_raw),
        "raw_positive_ratio": float(raw_positive_ratio) if not pd.isna(raw_positive_ratio) else np.nan,
        "clean_positive_ratio": float(clean_positive_ratio) if not pd.isna(clean_positive_ratio) else np.nan,
        "class_balance_shift": (
            float(abs(clean_positive_ratio - raw_positive_ratio))
            if not pd.isna(raw_positive_ratio) and not pd.isna(clean_positive_ratio)
            else np.nan
        ),
    }


def compute_cleaning_score(
    f1_macro: float,
    retention_rate: float,
    class_balance_shift: float,
    retention_penalty: float = 0.10,
    imbalance_penalty: float = 0.05,
) -> float:
    """
    Compute a penalized cleaning score.

    This is useful for sweet-spot threshold selection.

    Parameters
    ----------
    f1_macro : float
        Validation macro F1-score.
    retention_rate : float
        Fraction of data retained after cleaning.
    class_balance_shift : float
        Absolute shift in positive-class ratio after cleaning.
    retention_penalty : float
        Penalty weight for removed data.
    imbalance_penalty : float
        Penalty weight for class-balance shift.

    Returns
    -------
    float
        Penalized cleaning score.
    """

    if pd.isna(f1_macro):
        return np.nan

    if pd.isna(retention_rate):
        retention_rate = 0.0

    if pd.isna(class_balance_shift):
        class_balance_shift = 0.0

    return float(
        f1_macro
        - retention_penalty * (1.0 - retention_rate)
        - imbalance_penalty * class_balance_shift
    )


def export_clean_image_list(
    clean_df: pd.DataFrame,
    output_path: str | Path,
    columns: list[str] | None = None,
) -> Path:
    """
    Export cleaned image list to CSV.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Clean dataframe.
    output_path : str | Path
        Output CSV path.
    columns : list[str] | None
        Columns to export. If None, export path, label, and label_name.

    Returns
    -------
    Path
        Saved CSV path.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = [c for c in ["path", "label", "label_name"] if c in clean_df.columns]

    clean_df[columns].to_csv(output_path, index=False)

    return output_path


def make_cleaning_config_summary(cleaning_config: dict[str, Any]) -> pd.DataFrame:
    """
    Convert cleaning configuration to a display-friendly dataframe.

    Parameters
    ----------
    cleaning_config : dict
        Cleaning configuration.

    Returns
    -------
    pd.DataFrame
        Configuration table.
    """

    return pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in cleaning_config.items()]
    )