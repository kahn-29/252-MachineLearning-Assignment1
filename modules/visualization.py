from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def savefig(path: str | Path, dpi: int = 160, bbox_inches: str = "tight") -> Path:
    """Save the current matplotlib figure and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches=bbox_inches)
    return output


def plot_sample_grid(
    df,
    n_per_class=5,
    path_col="path",
    label_col="label_name",
    class_order=None,
    seed=42,
    figsize=None,
    save_path=None,
):
    """Plot sampled images by class."""

    if class_order is None:
        class_order = sorted(df[label_col].dropna().unique().tolist())

    if len(class_order) == 0:
        return

    if figsize is None:
        figsize = (n_per_class * 2.4, len(class_order) * 2.4)

    fig, axes = plt.subplots(len(class_order), n_per_class, figsize=figsize)

    if len(class_order) == 1:
        axes = np.array([axes])
    axes = np.asarray(axes)

    for row_idx, class_name in enumerate(class_order):
        group = df[df[label_col] == class_name]
        if len(group) == 0:
            continue

        sample = group.sample(n=min(n_per_class, len(group)), random_state=seed)

        for col_idx in range(n_per_class):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(sample):
                ax.axis("off")
                continue

            path = sample.iloc[col_idx][path_col]
            try:
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")

            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(str(class_name), fontsize=11, fontweight="bold")

    plt.suptitle("Sample Images by Class", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_class_distribution_pie(
    df,
    label_col="label_name",
    title="Class Distribution",
    save_path=None,
):
    """Plot class distribution as a pie chart with percentages."""

    counts = df[label_col].value_counts().sort_index()
    if counts.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title(title, fontweight="bold")
    plt.axis("equal")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_class_distribution_bar(
    df,
    label_col="label_name",
    title="Class Distribution",
    save_path=None,
):
    """Plot class distribution as a bar chart."""

    counts = df[label_col].value_counts().sort_index()
    if counts.empty:
        return

    total = counts.sum()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(counts.index.astype(str), counts.values)

    for bar, count in zip(bars, counts.values):
        pct = (count / total) * 100.0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_image_size_distribution(
    audit_df,
    width_col="width",
    height_col="height",
    label_col="label_name",
    title="Raw Image Size Distribution",
    save_path=None,
):
    """Plot width-height scatter distribution."""

    plot_df = audit_df.dropna(subset=[width_col, height_col]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=width_col,
        y=height_col,
        hue=label_col if label_col in plot_df.columns else None,
        alpha=0.45,
        s=28,
    )
    plt.title(title, fontweight="bold")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(axis="both", linestyle="--", alpha=0.3)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_rgb_channel_kde(
    df,
    sample_per_class=300,
    path_col="path",
    label_col="label_name",
    class_order=None,
    seed=42,
    save_path=None,
):
    """Plot KDE of per-image mean RGB channels by class."""

    if class_order is None:
        class_order = sorted(df[label_col].dropna().unique().tolist())

    if len(class_order) == 0:
        return

    sampled_parts = []
    for class_name in class_order:
        group = df[df[label_col] == class_name]
        if len(group) == 0:
            continue
        sampled_parts.append(group.sample(n=min(sample_per_class, len(group)), random_state=seed))

    if not sampled_parts:
        return

    sampled = pd.concat(sampled_parts, ignore_index=True)

    def mean_rgb(path: str) -> tuple[float, float, float]:
        try:
            arr = np.asarray(Image.open(path).convert("RGB").resize((64, 64)))
            return float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean())
        except Exception:
            return np.nan, np.nan, np.nan

    sampled[["R_mean", "G_mean", "B_mean"]] = pd.DataFrame(sampled[path_col].apply(mean_rgb).tolist(), index=sampled.index)

    fig, axes = plt.subplots(1, len(class_order), figsize=(6 * len(class_order), 4.5))
    if len(class_order) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, class_order):
        part = sampled[sampled[label_col] == class_name].dropna(subset=["R_mean", "G_mean", "B_mean"])
        if part.empty:
            ax.axis("off")
            continue

        for col, color in [("R_mean", "red"), ("G_mean", "green"), ("B_mean", "blue")]:
            sns.kdeplot(data=part, x=col, color=color, fill=True, alpha=0.18, ax=ax, label=color.upper())

        ax.set_title(str(class_name), fontweight="bold")
        ax.set_xlabel("Mean Intensity")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.suptitle("RGB Channel KDE by Class", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_metric_distribution(
    audit_df,
    metric: str,
    label_col="label_name",
    thresholds=None,
    title=None,
    save_path=None,
):
    """Plot one metric distribution, optionally with threshold lines."""

    if metric not in audit_df.columns:
        raise KeyError(f"Metric not found: {metric}")

    plot_df = audit_df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=plot_df,
        x=metric,
        hue=label_col if label_col in plot_df.columns else None,
        fill=True,
        common_norm=False,
        alpha=0.2,
    )

    if thresholds is not None:
        for value in thresholds:
            plt.axvline(value, linestyle="--", linewidth=1.2)

    plt.title(title or f"Distribution of {metric}", fontweight="bold")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_metric_correlation_heatmap(
    audit_df,
    metrics: list[str],
    title="Correlation Matrix of Image Quality Metrics",
    save_path=None,
):
    """Plot correlation heatmap of selected audit metrics."""

    cols = [col for col in metrics if col in audit_df.columns]
    if not cols:
        raise ValueError("No valid metric columns were provided.")

    corr = audit_df[cols].corr()

    plt.figure(figsize=(max(8, len(cols) * 0.75), max(6, len(cols) * 0.65)))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.5)
    plt.title(title, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_before_after_cleaning(summary_df, save_path=None):
    """Plot class counts across before_cleaning/after_cleaning/removed groups."""

    if "group" not in summary_df.columns:
        raise KeyError("summary_df must include a 'group' column.")

    value_cols = [c for c in summary_df.columns if c not in {"group", "total", "pct_of_original"} and not str(c).endswith("_pct")]
    if not value_cols:
        raise ValueError("No class-count columns found in summary_df.")

    melted = summary_df.melt(id_vars="group", value_vars=value_cols, var_name="class", value_name="count")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="group", y="count", hue="class")
    plt.title("Class Counts Before and After Cleaning", fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_removed_examples(
    removed_df,
    reason=None,
    n=8,
    path_col="path",
    label_col="label_name",
    reason_col="removal_reason",
    seed=42,
    save_path=None,
):
    """Plot removed-image examples, optionally filtered by reason substring."""

    view_df = removed_df.copy()
    if reason is not None and reason_col in view_df.columns:
        view_df = view_df[view_df[reason_col].astype(str).str.contains(str(reason), regex=False, na=False)]

    if view_df.empty:
        return

    sample = view_df.sample(n=min(n, len(view_df)), random_state=seed)

    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.4))
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        try:
            img = Image.open(row[path_col]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")

        ax.axis("off")

        title_parts = []
        if label_col in row:
            title_parts.append(str(row[label_col]))
        if reason_col in row:
            title_parts.append(str(row[reason_col]))
        ax.set_title("\n".join(title_parts), fontsize=8)

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("Removed Image Examples", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_split_distribution(
    train_df,
    val_df,
    test_df,
    label_col="label_name",
    save_path=None,
):
    """Plot class distribution for train/validation/test splits."""

    records: list[dict[str, Any]] = []

    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = split_df[label_col].value_counts()
        for class_name, count in counts.items():
            records.append({"split": split_name, "class": class_name, "count": int(count)})

    if not records:
        return

    plot_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="split", y="count", hue="class")
    plt.title("Class Distribution Across Splits", fontweight="bold")
    plt.xlabel("Split")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_transform_examples(transform_records, n_images=None, save_path=None):
    """Plot prepared transform preview records containing path/mode/image."""

    if not transform_records:
        return

    df = pd.DataFrame(transform_records)
    if df.empty or not {"path", "mode", "image"}.issubset(df.columns):
        return

    paths = df["path"].drop_duplicates().tolist()
    if n_images is not None:
        paths = paths[: int(n_images)]
    modes = df["mode"].drop_duplicates().tolist()

    if not paths or not modes:
        return

    fig, axes = plt.subplots(len(paths), len(modes), figsize=(len(modes) * 3.0, len(paths) * 3.0))

    if len(paths) == 1 and len(modes) == 1:
        axes = np.array([[axes]])
    elif len(paths) == 1:
        axes = np.array([axes])
    elif len(modes) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, path in enumerate(paths):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            sub = df[(df["path"] == path) & (df["mode"] == mode)]
            if sub.empty:
                ax.axis("off")
                continue

            image = sub.iloc[0]["image"]
            ax.imshow(image)
            ax.axis("off")
            if i == 0:
                ax.set_title(str(mode), fontweight="bold")

    plt.suptitle("Transform Examples", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_confusion_matrix(
    cm,
    labels=("Cat", "Dog"),
    title="Confusion Matrix",
    save_path=None,
):
    """Plot confusion matrix heatmap from a numeric matrix."""

    matrix = np.asarray(cm)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
    )
    plt.title(title, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_wrong_predictions(
    wrong_df,
    n=8,
    path_col="path",
    true_col="true_label_name",
    pred_col="pred_label_name",
    save_path=None,
):
    """Plot misclassified examples with true and predicted labels."""

    if wrong_df.empty:
        return

    sample = wrong_df.head(n)
    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.4))
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        try:
            img = Image.open(row[path_col]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")

        ax.axis("off")
        true_label = row.get(true_col, row.get("true_label", "?"))
        pred_label = row.get(pred_col, row.get("pred_label", "?"))
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("Misclassified Samples", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_training_curves(history_df, save_path_prefix=None, epoch_col=None):
    """Plot train/validation accuracy and loss curves from history dataframe."""

    x = history_df[epoch_col] if epoch_col is not None and epoch_col in history_df.columns else np.arange(1, len(history_df) + 1)

    if {"train_accuracy", "val_accuracy"}.issubset(history_df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(x, history_df["train_accuracy"], marker="o", label="Train Accuracy")
        plt.plot(x, history_df["val_accuracy"], marker="o", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy", fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        if save_path_prefix is not None:
            savefig(f"{save_path_prefix}_accuracy.png")
        plt.show()

    if {"train_loss", "val_loss"}.issubset(history_df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(x, history_df["train_loss"], marker="o", label="Train Loss")
        plt.plot(x, history_df["val_loss"], marker="o", label="Validation Loss")
        plt.title("Training and Validation Loss", fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        if save_path_prefix is not None:
            savefig(f"{save_path_prefix}_loss.png")
        plt.show()


def plot_model_comparison(
    comparison_df,
    metric="f1_macro",
    model_col="model",
    title=None,
    save_path=None,
):
    """Plot model comparison dataframe; skip when a single model is present."""

    if len(comparison_df) <= 1:
        return

    if metric not in comparison_df.columns:
        raise KeyError(f"Metric column not found: {metric}")
    if model_col not in comparison_df.columns:
        raise KeyError(f"Model column not found: {model_col}")

    view_df = comparison_df.sort_values(metric, ascending=False)

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(view_df[model_col].astype(str), view_df[metric])

    for bar, value in zip(bars, view_df[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            float(value),
            f"{float(value):.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title or f"Model Comparison by {metric}", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()
