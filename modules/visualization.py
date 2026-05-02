# modules/visualization.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def savefig(
    path: str | Path,
    dpi: int = 160,
    bbox_inches: str = "tight",
) -> Path:
    """
    Save current matplotlib figure.

    Parameters
    ----------
    path : str | Path
        Output figure path.
    dpi : int
        Figure resolution.
    bbox_inches : str
        Bounding box option for matplotlib.

    Returns
    -------
    Path
        Saved figure path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

    return path


def plot_sample_grid(
    df: pd.DataFrame,
    n_per_class: int = 5,
    path_col: str = "path",
    label_col: str = "label_name",
    class_order: list[str] | None = None,
    seed: int = 42,
    figsize: tuple[int, int] | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot sample images by class.

    For binary Cat/Dog setting, this can create one row for Cat
    and one row for Dog.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    n_per_class : int
        Number of images per class.
    path_col : str
        Column containing image paths.
    label_col : str
        Column containing class names.
    class_order : list[str] | None
        Order of classes. If None, sorted unique labels are used.
    seed : int
        Random seed for sampling.
    figsize : tuple[int, int] | None
        Figure size.
    save_path : str | Path | None
        Optional output path.
    """

    if class_order is None:
        class_order = sorted(df[label_col].dropna().unique().tolist())

    if figsize is None:
        figsize = (n_per_class * 2.4, len(class_order) * 2.4)

    fig, axes = plt.subplots(
        len(class_order),
        n_per_class,
        figsize=figsize,
    )

    if len(class_order) == 1:
        axes = np.array([axes])

    axes = np.asarray(axes)

    for row_idx, class_name in enumerate(class_order):
        sub = df[df[label_col] == class_name]

        if len(sub) == 0:
            continue

        sample = sub.sample(
            n=min(n_per_class, len(sub)),
            random_state=seed,
        )

        for col_idx in range(n_per_class):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(sample):
                ax.axis("off")
                continue

            img_path = sample.iloc[col_idx][path_col]

            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")

            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(
                    class_name,
                    fontsize=12,
                    fontweight="bold",
                )

    plt.suptitle("Sample Images by Class", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_class_distribution_pie(
    df: pd.DataFrame,
    label_col: str = "label_name",
    title: str = "Class Distribution",
    save_path: str | Path | None = None,
):
    """
    Plot class distribution as a pie chart with count and percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Class label column.
    title : str
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()

    labels = [
        f"{label}: {count} ({count / total * 100:.1f}%)"
        for label, count in counts.items()
    ]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        startangle=90,
        autopct="%1.1f%%",
    )
    plt.legend(
        labels,
        title="Class",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    plt.title(title, fontweight="bold")
    plt.axis("equal")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_class_distribution_bar(
    df: pd.DataFrame,
    label_col: str = "label_name",
    title: str = "Class Distribution",
    save_path: str | Path | None = None,
):
    """
    Plot class distribution as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Class label column.
    title : str
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(counts.index, counts.values)

    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title(title, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_image_size_distribution(
    audit_df: pd.DataFrame,
    width_col: str = "width",
    height_col: str = "height",
    label_col: str = "label_name",
    title: str = "Raw Image Size Distribution",
    save_path: str | Path | None = None,
):
    """
    Plot image width-height distribution.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    width_col : str
        Width column.
    height_col : str
        Height column.
    label_col : str
        Class label column.
    title : str
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    plot_df = audit_df.dropna(subset=[width_col, height_col]).copy()

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
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.grid(axis="both", linestyle="--", alpha=0.3)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_rgb_channel_kde(
    df: pd.DataFrame,
    sample_per_class: int = 300,
    path_col: str = "path",
    label_col: str = "label_name",
    class_order: list[str] | None = None,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """
    Plot KDE distribution of mean RGB channel intensity by class.

    This plot uses one mean R/G/B value per image, making it more readable
    than pixel-level histograms.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sample_per_class : int
        Number of images sampled per class.
    path_col : str
        Image path column.
    label_col : str
        Class label column.
    class_order : list[str] | None
        Class order.
    seed : int
        Random seed.
    save_path : str | Path | None
        Optional output path.
    """

    if class_order is None:
        class_order = sorted(df[label_col].dropna().unique().tolist())

    sampled_parts = []

    for class_name in class_order:
        sub = df[df[label_col] == class_name]

        if len(sub) == 0:
            continue

        sampled_parts.append(
            sub.sample(
                n=min(sample_per_class, len(sub)),
                random_state=seed,
            )
        )

    df_sample = pd.concat(sampled_parts, ignore_index=True)

    def compute_rgb_mean(path):
        try:
            img = Image.open(path).convert("RGB").resize((64, 64))
            arr = np.asarray(img)
            return (
                arr[:, :, 0].mean(),
                arr[:, :, 1].mean(),
                arr[:, :, 2].mean(),
            )
        except Exception:
            return (np.nan, np.nan, np.nan)

    rgb_means = df_sample[path_col].apply(compute_rgb_mean)
    df_sample[["R_mean", "G_mean", "B_mean"]] = pd.DataFrame(
        rgb_means.tolist(),
        index=df_sample.index,
    )

    fig, axes = plt.subplots(
        1,
        len(class_order),
        figsize=(7 * len(class_order), 5),
    )

    if len(class_order) == 1:
        axes = [axes]

    channel_cfg = [
        ("R_mean", "red"),
        ("G_mean", "green"),
        ("B_mean", "blue"),
    ]

    for ax, class_name in zip(axes, class_order):
        sub = df_sample[df_sample[label_col] == class_name].dropna(
            subset=["R_mean", "G_mean", "B_mean"]
        )

        for col, color in channel_cfg:
            sns.kdeplot(
                data=sub,
                x=col,
                color=color,
                label=color.capitalize(),
                fill=True,
                alpha=0.15,
                ax=ax,
            )

        ax.set_title(
            f"RGB Channel Distribution — {class_name.upper()}",
            fontweight="bold",
        )
        ax.set_xlabel("Mean Pixel Intensity (0–255)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(
        "Per-Channel Colour Distribution by Class",
        fontsize=14,
        fontweight="bold",
    )

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_metric_distribution(
    audit_df: pd.DataFrame,
    metric: str,
    label_col: str = "label_name",
    thresholds: list[float] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot distribution of one audit metric by class.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    metric : str
        Metric column.
    label_col : str
        Class label column.
    thresholds : list[float] | None
        Optional threshold lines.
    title : str | None
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    if metric not in audit_df.columns:
        raise KeyError(f"Metric not found: {metric}")

    plot_df = audit_df.dropna(subset=[metric]).copy()

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
        for t in thresholds:
            plt.axvline(
                t,
                linestyle="--",
                linewidth=1.2,
                label=f"Threshold = {t}",
            )

    plt.title(title or f"Distribution of {metric}", fontweight="bold")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if thresholds is not None:
        plt.legend()

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_metric_correlation_heatmap(
    audit_df: pd.DataFrame,
    metrics: list[str],
    title: str = "Correlation Matrix of Image Quality Metrics",
    save_path: str | Path | None = None,
):
    """
    Plot correlation heatmap of selected audit metrics.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Audit dataframe.
    metrics : list[str]
        Metric columns.
    title : str
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    cols = [m for m in metrics if m in audit_df.columns]

    if len(cols) == 0:
        raise ValueError("No valid metric columns provided.")

    corr = audit_df[cols].corr()

    plt.figure(figsize=(max(10, len(cols) * 0.75), max(8, len(cols) * 0.65)))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    plt.title(title, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_removed_examples(
    removed_df: pd.DataFrame,
    reason: str | None = None,
    n: int = 8,
    path_col: str = "path",
    label_col: str = "label_name",
    reason_col: str = "removal_reason",
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """
    Plot examples of removed images.

    Parameters
    ----------
    removed_df : pd.DataFrame
        Removed image dataframe.
    reason : str | None
        Optional reason filter.
    n : int
        Number of images to show.
    path_col : str
        Image path column.
    label_col : str
        Label name column.
    reason_col : str
        Removal reason column.
    seed : int
        Random seed.
    save_path : str | Path | None
        Optional output path.
    """

    df = removed_df.copy()

    if reason is not None and reason_col in df.columns:
        df = df[df[reason_col].astype(str).str.contains(reason, regex=False, na=False)]

    if len(df) == 0:
        print("No removed examples to display.")
        return

    sample = df.sample(n=min(n, len(df)), random_state=seed)

    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.5))

    if rows == 1 and cols == 1:
        axes = np.array([axes])

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

    title = "Removed Image Examples"
    if reason is not None:
        title += f" — {reason}"

    plt.suptitle(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_before_after_cleaning(
    summary_df: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Plot class counts before and after cleaning.

    Expected summary_df columns:
    group, cat, dog.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Cleaning summary dataframe.
    save_path : str | Path | None
        Optional output path.
    """

    required = {"group", "cat", "dog"}

    if not required.issubset(summary_df.columns):
        raise KeyError(f"summary_df must contain columns: {required}")

    plot_df = summary_df.melt(
        id_vars="group",
        value_vars=["cat", "dog"],
        var_name="class",
        value_name="count",
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="group",
        y="count",
        hue="class",
    )
    plt.title("Class Counts Before and After Cleaning", fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Number of images")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label_name",
    save_path: str | Path | None = None,
):
    """
    Plot class distribution across train/validation/test splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    val_df : pd.DataFrame
        Validation dataframe.
    test_df : pd.DataFrame
        Testing dataframe.
    label_col : str
        Label column.
    save_path : str | Path | None
        Optional output path.
    """

    records = []

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        counts = split_df[label_col].value_counts()

        for label, count in counts.items():
            records.append(
                {
                    "split": split_name,
                    "class": label,
                    "count": count,
                }
            )

    plot_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="split",
        y="count",
        hue="class",
    )

    plt.title("Class Distribution Across Data Splits", fontweight="bold")
    plt.xlabel("Split")
    plt.ylabel("Number of images")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_transform_examples(
    transform_records: list[dict[str, Any]],
    n_images: int | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot transformation preview records.

    Expected records:
    [
        {"path": ..., "mode": ..., "image": PIL.Image},
        ...
    ]

    Parameters
    ----------
    transform_records : list[dict]
        Output from transforms.get_transform_grid().
    n_images : int | None
        Number of original images to display.
    save_path : str | Path | None
        Optional output path.
    """

    if len(transform_records) == 0:
        print("No transform records to display.")
        return

    df = pd.DataFrame(transform_records)

    paths = df["path"].drop_duplicates().tolist()

    if n_images is not None:
        paths = paths[:n_images]

    modes = df["mode"].drop_duplicates().tolist()

    fig, axes = plt.subplots(
        len(paths),
        len(modes),
        figsize=(len(modes) * 3.0, len(paths) * 3.0),
    )

    if len(paths) == 1 and len(modes) == 1:
        axes = np.array([[axes]])
    elif len(paths) == 1:
        axes = np.array([axes])
    elif len(modes) == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, path in enumerate(paths):
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx, col_idx]
            sub = df[(df["path"] == path) & (df["mode"] == mode)]

            if len(sub) == 0:
                ax.axis("off")
                continue

            img = sub.iloc[0]["image"]
            ax.imshow(img)
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(mode, fontweight="bold")

    plt.suptitle("Image Transformation Examples", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_grid_search_topk(
    result_df: pd.DataFrame,
    metric: str = "accuracy",
    top_k: int = 10,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot top-k grid-search configurations.

    Parameters
    ----------
    result_df : pd.DataFrame
        Grid-search result dataframe.
    metric : str
        Metric column.
    top_k : int
        Number of configurations to show.
    title : str | None
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    if metric not in result_df.columns:
        raise KeyError(f"Metric column not found: {metric}")

    df = result_df.sort_values(metric, ascending=False).head(top_k).copy()

    label_cols = [
        c for c in ["Transform", "Preprocessing", "Backbone", "Classifier", "model"]
        if c in df.columns
    ]

    if len(label_cols) == 0:
        df["config"] = df.index.astype(str)
    else:
        df["config"] = df[label_cols].astype(str).agg(" + ".join, axis=1)

    plt.figure(figsize=(10, max(5, top_k * 0.45)))
    sns.barplot(
        data=df,
        x=metric,
        y="config",
        orient="h",
    )

    for i, value in enumerate(df[metric]):
        plt.text(
            value,
            i,
            f" {value:.4f}",
            va="center",
            fontsize=9,
        )

    plt.title(title or f"Top {top_k} Configurations by {metric}", fontweight="bold")
    plt.xlabel(metric)
    plt.ylabel("Configuration")
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_grid_search_heatmap(
    result_df: pd.DataFrame,
    row: str = "Transform",
    col: str = "Backbone",
    value: str = "Accuracy",
    aggfunc: str = "max",
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot heatmap of grid-search performance.

    Parameters
    ----------
    result_df : pd.DataFrame
        Grid-search result dataframe.
    row : str
        Row variable.
    col : str
        Column variable.
    value : str
        Metric value.
    aggfunc : str
        Aggregation function.
    title : str | None
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    for c in [row, col, value]:
        if c not in result_df.columns:
            raise KeyError(f"Column not found in result_df: {c}")

    pivot = result_df.pivot_table(
        index=row,
        columns=col,
        values=value,
        aggfunc=aggfunc,
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        linewidths=0.5,
    )

    plt.title(
        title or f"Grid Search Heatmap ({aggfunc} {value})",
        fontweight="bold",
    )
    plt.xlabel(col)
    plt.ylabel(row)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_confusion_matrix(
    cm,
    labels=("Cat", "Dog"),
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
):
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    cm : array-like
        Confusion matrix.
    labels : tuple
        Class labels.
    title : str
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    cm = np.asarray(cm)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_wrong_predictions(
    wrong_df: pd.DataFrame,
    n: int = 8,
    path_col: str = "path",
    true_col: str = "true_label_name",
    pred_col: str = "pred_label_name",
    save_path: str | Path | None = None,
):
    """
    Plot wrong prediction cases.

    Parameters
    ----------
    wrong_df : pd.DataFrame
        Dataframe of wrong predictions.
    n : int
        Number of images to display.
    path_col : str
        Image path column.
    true_col : str
        True label name column.
    pred_col : str
        Predicted label name column.
    save_path : str | Path | None
        Optional output path.
    """

    if len(wrong_df) == 0:
        print("No wrong predictions to display.")
        return

    sample = wrong_df.head(n)

    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.4))

    if rows == 1 and cols == 1:
        axes = np.array([axes])

    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        try:
            img = Image.open(row[path_col]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")

        ax.axis("off")

        true_label = row[true_col] if true_col in row else row.get("true_label", "?")
        pred_label = row[pred_col] if pred_col in row else row.get("pred_label", "?")

        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}",
            fontsize=9,
        )

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("Misclassified Samples", fontsize=14, fontweight="bold")

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_training_curves(
    history_df: pd.DataFrame,
    save_path_prefix: str | Path | None = None,
    epoch_col: str | None = None,
):
    """
    Plot training and validation accuracy/loss curves.

    Expected columns:
    train_accuracy, val_accuracy, train_loss, val_loss.

    Parameters
    ----------
    history_df : pd.DataFrame
        Training history dataframe.
    save_path_prefix : str | Path | None
        Prefix for saving figures.
    epoch_col : str | None
        Optional epoch column.
    """

    if epoch_col is not None and epoch_col in history_df.columns:
        x = history_df[epoch_col]
    else:
        x = np.arange(1, len(history_df) + 1)

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
    comparison_df: pd.DataFrame,
    metric: str = "f1_macro",
    model_col: str = "model",
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot model comparison bar chart.

    This should only be used when there is more than one model.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison dataframe.
    metric : str
        Metric column.
    model_col : str
        Model name column.
    title : str | None
        Figure title.
    save_path : str | Path | None
        Optional output path.
    """

    if len(comparison_df) <= 1:
        print("Single model detected. Model comparison chart skipped.")
        return

    if metric not in comparison_df.columns:
        raise KeyError(f"Metric column not found: {metric}")

    if model_col not in comparison_df.columns:
        raise KeyError(f"Model column not found: {model_col}")

    df = comparison_df.sort_values(metric, ascending=False)

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(df[model_col], df[metric])

    ymin = max(0.0, df[metric].min() - 0.02)
    ymax = min(1.0, df[metric].max() + 0.01)
    plt.ylim(ymin, ymax)

    for bar, value in zip(bars, df[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
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


def plot_cleaning_preset_comparison(
    preset_df: pd.DataFrame,
    metric: str = "f1_macro",
    preset_col: str = "filter",
    baseline_value: float | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot comparison of cleaning presets.

    Parameters
    ----------
    preset_df : pd.DataFrame
        Cleaning preset result dataframe.
    metric : str
        Metric column.
    preset_col : str
        Preset name column.
    baseline_value : float | None
        Optional baseline reference line.
    save_path : str | Path | None
        Optional output path.
    """

    if metric not in preset_df.columns:
        raise KeyError(f"Metric not found: {metric}")

    if preset_col not in preset_df.columns:
        raise KeyError(f"Preset column not found: {preset_col}")

    df = preset_df.sort_values(metric, ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df,
        x=preset_col,
        y=metric,
    )

    if baseline_value is not None:
        plt.axhline(
            baseline_value,
            linestyle="--",
            color="black",
            label="Baseline",
        )
        plt.legend()

    plt.title("Cleaning Preset Comparison", fontweight="bold")
    plt.xlabel("Cleaning preset")
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if save_path is not None:
        savefig(save_path)

    plt.show()


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    x_col: str = "retention_pct",
    y_col: str = "f1_macro",
    hue_col: str = "filter",
    baseline_value: float | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot threshold sweep result.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Threshold sweep dataframe.
    x_col : str
        X-axis column.
    y_col : str
        Y-axis column.
    hue_col : str
        Hue column.
    baseline_value : float | None
        Optional baseline y-value.
    save_path : str | Path | None
        Optional output path.
    """

    for col in [x_col, y_col]:
        if col not in sweep_df.columns:
            raise KeyError(f"Column not found: {col}")

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=sweep_df,
        x=x_col,
        y=y_col,
        hue=hue_col if hue_col in sweep_df.columns else None,
        s=90,
    )

    if baseline_value is not None:
        plt.axhline(
            baseline_value,
            linestyle="--",
            color="black",
            label="Baseline",
        )

    plt.title("Threshold Sweep Result", fontweight="bold")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(axis="both", linestyle="--", alpha=0.3)

    if hue_col in sweep_df.columns:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if save_path is not None:
        savefig(save_path)

    plt.show()