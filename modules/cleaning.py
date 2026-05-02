from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


class _UnionFind:
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


def _hamming_distance_hex(hash_a: str | None, hash_b: str | None) -> int:
    if hash_a is None or hash_b is None:
        return 10**9
    try:
        return (int(str(hash_a), 16) ^ int(str(hash_b), 16)).bit_count()
    except Exception:
        return 10**9


def _rank_score(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)

    values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=df.index)

    return values.rank(pct=True, ascending=ascending).fillna(0.0)


def compute_quality_score(audit_df: pd.DataFrame) -> pd.Series:
    """Compute a relative quality score used for duplicate representative choice."""

    df = audit_df.copy()
    score = (
        _rank_score(df, "blur_laplacian", ascending=True)
        + _rank_score(df, "entropy", ascending=True)
        + _rank_score(df, "min_side", ascending=True)
        + _rank_score(df, "brightness_std", ascending=True)
        + _rank_score(df, "center_saliency_ratio", ascending=True)
        - _rank_score(df, "compression_artifact", ascending=True)
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
    """Detect near-duplicates from perceptual hashes and keep best representative."""

    if hash_col not in audit_df.columns:
        raise KeyError(f"Hash column not found: {hash_col}")
    if bands <= 0:
        raise ValueError("bands must be a positive integer.")

    df = audit_df.copy().reset_index(drop=True)
    n = len(df)

    hashes = df[hash_col].astype(object).where(df[hash_col].notna(), None).tolist()
    valid_hash = np.array([isinstance(h, str) and len(h) > 0 for h in hashes])

    uf = _UnionFind(n)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    bits_per_band = max(1, 64 // bands)

    for i, hash_value in enumerate(hashes):
        if not valid_hash[i]:
            continue
        try:
            value = int(hash_value, 16)
        except Exception:
            continue

        for band_idx in range(bands):
            shift = band_idx * bits_per_band
            band_value = (value >> shift) & ((1 << bits_per_band) - 1)
            buckets[(band_idx, band_value)].append(i)

    candidate_pairs: set[tuple[int, int]] = set()
    for idx_list in buckets.values():
        if len(idx_list) < 2:
            continue
        for a, b in combinations(idx_list, 2):
            candidate_pairs.add((min(a, b), max(a, b)))

    for a, b in candidate_pairs:
        if _hamming_distance_hex(hashes[a], hashes[b]) <= hamming_threshold:
            uf.union(a, b)

    clusters = np.array([uf.find(i) for i in range(n)])
    quality = compute_quality_score(df)

    representatives = np.zeros(n, dtype=bool)
    group_df = pd.DataFrame(
        {
            "idx": np.arange(n),
            "cluster": clusters,
            "quality": quality.values,
            "has_valid_hash": valid_hash,
        }
    )

    for _, group in group_df.groupby("cluster"):
        if len(group) == 1:
            representatives[int(group["idx"].iloc[0])] = True
            continue

        valid = group[group["has_valid_hash"]]
        if len(valid) == 0:
            representatives[int(group["idx"].iloc[0])] = True
            continue

        best_idx = int(valid.sort_values("quality", ascending=False)["idx"].iloc[0])
        representatives[best_idx] = True

    size_map = pd.Series(clusters).value_counts().to_dict()
    group_sizes = np.array([size_map[c] for c in clusters])

    df["duplicate_cluster"] = clusters
    df["duplicate_group_size"] = group_sizes
    df["is_duplicate_representative"] = representatives
    df["is_duplicate"] = (group_sizes > 1) & (~representatives)

    return df


def compute_soft_flags(audit_df: pd.DataFrame, cleaning_config: dict) -> pd.DataFrame:
    """Compute warning flags and soft_flag_count from configured thresholds."""

    df = audit_df.copy()
    flags = pd.DataFrame(index=df.index)

    def lower(col: str, key: str, flag_name: str) -> None:
        threshold = cleaning_config.get(key)
        if threshold is not None and col in df.columns:
            flags[flag_name] = pd.to_numeric(df[col], errors="coerce") < threshold
        else:
            flags[flag_name] = False

    def upper(col: str, key: str, flag_name: str) -> None:
        threshold = cleaning_config.get(key)
        if threshold is not None and col in df.columns:
            flags[flag_name] = pd.to_numeric(df[col], errors="coerce") > threshold
        else:
            flags[flag_name] = False

    lower("entropy", "entropy_min", "soft_low_entropy")
    lower("mean_sat", "mean_sat_min", "soft_low_saturation")
    lower("center_saliency_ratio", "center_saliency_ratio_min", "soft_low_saliency")
    lower("brightness_std", "brightness_std_min", "soft_low_brightness_std")
    lower("chroma_mean", "chroma_mean_min", "soft_low_chroma")

    upper("compression_artifact", "compression_artifact_max", "soft_high_compression_artifact")
    upper("brightness_std", "brightness_std_max", "soft_high_brightness_std")

    flags = flags.fillna(False).astype(bool)
    flags["soft_flag_count"] = flags.sum(axis=1)

    return flags


def build_cleaning_mask(audit_df: pd.DataFrame, cleaning_config: dict) -> pd.Series:
    """Build keep/remove boolean mask where True means kept and False means removed."""

    df = audit_df.copy()
    keep = pd.Series(True, index=df.index)

    if cleaning_config.get("remove_corrupted", True) and "is_corrupted" in df.columns:
        keep &= ~df["is_corrupted"].fillna(False).astype(bool)

    if cleaning_config.get("remove_duplicates", True) and "is_duplicate" in df.columns:
        keep &= ~df["is_duplicate"].fillna(False).astype(bool)

    if cleaning_config.get("blur_laplacian_min") is not None and "blur_laplacian" in df.columns:
        keep &= pd.to_numeric(df["blur_laplacian"], errors="coerce") >= cleaning_config["blur_laplacian_min"]

    if cleaning_config.get("min_side_min") is not None and "min_side" in df.columns:
        keep &= pd.to_numeric(df["min_side"], errors="coerce") >= cleaning_config["min_side_min"]

    if cleaning_config.get("aspect_extremity_max") is not None and "aspect_extremity" in df.columns:
        keep &= pd.to_numeric(df["aspect_extremity"], errors="coerce") <= cleaning_config["aspect_extremity_max"]

    if cleaning_config.get("near_mono_ratio_max") is not None and "near_mono_ratio" in df.columns:
        keep &= pd.to_numeric(df["near_mono_ratio"], errors="coerce") <= cleaning_config["near_mono_ratio_max"]

    entropy_near_mono_max = cleaning_config.get("entropy_near_mono_max")

    if cleaning_config.get("dark_ratio_max") is not None and "dark_ratio" in df.columns:
        dark_bad = pd.to_numeric(df["dark_ratio"], errors="coerce") > cleaning_config["dark_ratio_max"]
        if entropy_near_mono_max is not None and "entropy" in df.columns:
            dark_bad &= pd.to_numeric(df["entropy"], errors="coerce") < entropy_near_mono_max
        keep &= ~dark_bad.fillna(False)

    if cleaning_config.get("bright_ratio_max") is not None and "bright_ratio" in df.columns:
        bright_bad = pd.to_numeric(df["bright_ratio"], errors="coerce") > cleaning_config["bright_ratio_max"]
        if entropy_near_mono_max is not None and "entropy" in df.columns:
            bright_bad &= pd.to_numeric(df["entropy"], errors="coerce") < entropy_near_mono_max
        keep &= ~bright_bad.fillna(False)

    max_soft_flags = cleaning_config.get("max_soft_flags")
    if max_soft_flags is not None:
        soft_flags = compute_soft_flags(df, cleaning_config)
        keep &= soft_flags["soft_flag_count"] <= int(max_soft_flags)

    return keep.fillna(False).astype(bool)


def assign_removal_reasons(audit_df: pd.DataFrame, cleaning_config: dict) -> pd.DataFrame:
    """Add is_kept, soft_flag_count, and removal_reason columns."""

    df = audit_df.copy()
    keep = build_cleaning_mask(df, cleaning_config)
    soft = compute_soft_flags(df, cleaning_config)

    df["is_kept"] = keep.values
    df["soft_flag_count"] = soft["soft_flag_count"].astype(int).values

    reasons: list[str] = []
    for i, row in df.iterrows():
        if bool(keep.loc[i]):
            reasons.append("kept")
            continue

        row_reasons: list[str] = []

        if cleaning_config.get("remove_corrupted", True) and bool(row.get("is_corrupted", False)):
            row_reasons.append("corrupted")

        if cleaning_config.get("remove_duplicates", True) and bool(row.get("is_duplicate", False)):
            row_reasons.append("near_duplicate")

        if cleaning_config.get("blur_laplacian_min") is not None and row.get("blur_laplacian", np.inf) < cleaning_config["blur_laplacian_min"]:
            row_reasons.append("blurry")

        if cleaning_config.get("min_side_min") is not None and row.get("min_side", np.inf) < cleaning_config["min_side_min"]:
            row_reasons.append("undersized")

        if cleaning_config.get("aspect_extremity_max") is not None and row.get("aspect_extremity", 0) > cleaning_config["aspect_extremity_max"]:
            row_reasons.append("extreme_aspect_ratio")

        if cleaning_config.get("near_mono_ratio_max") is not None and row.get("near_mono_ratio", 0) > cleaning_config["near_mono_ratio_max"]:
            row_reasons.append("near_monochrome")

        entropy_near_mono_max = cleaning_config.get("entropy_near_mono_max")

        if cleaning_config.get("dark_ratio_max") is not None:
            dark_bad = row.get("dark_ratio", 0) > cleaning_config["dark_ratio_max"]
            if entropy_near_mono_max is not None:
                dark_bad = dark_bad and row.get("entropy", np.inf) < entropy_near_mono_max
            if dark_bad:
                row_reasons.append("near_mono_dark")

        if cleaning_config.get("bright_ratio_max") is not None:
            bright_bad = row.get("bright_ratio", 0) > cleaning_config["bright_ratio_max"]
            if entropy_near_mono_max is not None:
                bright_bad = bright_bad and row.get("entropy", np.inf) < entropy_near_mono_max
            if bright_bad:
                row_reasons.append("near_mono_bright")

        max_soft_flags = cleaning_config.get("max_soft_flags")
        if max_soft_flags is not None and int(soft.loc[i, "soft_flag_count"]) > int(max_soft_flags):
            row_reasons.append(f"too_many_soft_flags_{int(soft.loc[i, 'soft_flag_count'])}")

        reasons.append(", ".join(row_reasons) if row_reasons else "removed_by_combined_rule")

    df["removal_reason"] = reasons

    for col in soft.columns:
        if col != "soft_flag_count":
            df[col] = soft[col].values

    return df


def apply_cleaning(audit_df: pd.DataFrame, cleaning_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply cleaning rules and return kept and removed dataframes."""

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
    """Summarize total and class-level proportions before/after cleaning."""

    groups = [
        ("before_cleaning", raw_df),
        ("after_cleaning", clean_df),
        ("removed", removed_df),
    ]

    all_classes: list[str] = []
    for _, group_df in groups:
        if label_col in group_df.columns:
            all_classes.extend(group_df[label_col].dropna().astype(str).unique().tolist())
    classes = sorted(set(all_classes))

    records: list[dict] = []
    for group_name, group_df in groups:
        total = len(group_df)
        row = {"group": group_name, "total": int(total)}

        if label_col in group_df.columns and total > 0:
            counts = group_df[label_col].value_counts().to_dict()
        else:
            counts = {}

        for class_name in classes:
            count = int(counts.get(class_name, 0))
            key = str(class_name).lower().replace(" ", "_")
            row[key] = count
            row[f"{key}_pct"] = (count / total * 100.0) if total > 0 else 0.0

        records.append(row)

    summary = pd.DataFrame(records)
    original = max(int(summary.loc[summary["group"] == "before_cleaning", "total"].iloc[0]), 1)
    summary["pct_of_original"] = summary["total"] / original * 100.0

    return summary


def summarize_removal_reasons(
    removed_df: pd.DataFrame,
    reason_col: str = "removal_reason",
) -> pd.DataFrame:
    """Summarize removal reasons as count and percentage."""

    if len(removed_df) == 0:
        return pd.DataFrame(columns=["reason", "count", "percentage"])

    if reason_col not in removed_df.columns:
        raise KeyError(f"reason_col not found: {reason_col}")

    counter: dict[str, int] = defaultdict(int)
    for value in removed_df[reason_col].fillna("unknown"):
        for reason in str(value).split(","):
            reason = reason.strip()
            if reason:
                counter[reason] += 1

    summary = pd.DataFrame(
        [{"reason": reason, "count": count} for reason, count in counter.items()]
    ).sort_values("count", ascending=False)

    summary["percentage"] = summary["count"] / len(removed_df) * 100.0
    return summary.reset_index(drop=True)


def evaluate_cleaning_retention(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    label_col: str = "label",
) -> dict:
    """Compute retention/removal rates and class-balance shift."""

    n_raw = len(raw_df)
    n_clean = len(clean_df)

    if n_raw == 0:
        raise ValueError("raw_df is empty.")

    raw_ratio = pd.to_numeric(raw_df[label_col], errors="coerce").mean() if label_col in raw_df.columns else np.nan
    clean_ratio = pd.to_numeric(clean_df[label_col], errors="coerce").mean() if label_col in clean_df.columns and n_clean > 0 else np.nan

    if pd.isna(raw_ratio) or pd.isna(clean_ratio):
        balance_shift = np.nan
    else:
        balance_shift = abs(float(clean_ratio) - float(raw_ratio))

    return {
        "n_raw": int(n_raw),
        "n_clean": int(n_clean),
        "n_removed": int(n_raw - n_clean),
        "retention_rate": float(n_clean / n_raw),
        "removal_rate": float((n_raw - n_clean) / n_raw),
        "raw_positive_ratio": float(raw_ratio) if not pd.isna(raw_ratio) else np.nan,
        "clean_positive_ratio": float(clean_ratio) if not pd.isna(clean_ratio) else np.nan,
        "class_balance_shift": float(balance_shift) if not pd.isna(balance_shift) else np.nan,
    }
