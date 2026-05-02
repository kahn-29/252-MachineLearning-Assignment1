from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def savefig(path: str | Path, dpi: int = 160) -> None:
    """
    Save current matplotlib figure.
    """
    ...


def plot_sample_grid(
    df: pd.DataFrame,
    n_per_class: int = 5,
    path_col: str = "path",
    label_col: str = "label_name",
    save_path: str | Path | None = None
):
    """
    Plot sample images by class.
    Example: first row cats, second row dogs.
    """
    ...


def plot_class_distribution_pie(
    df: pd.DataFrame,
    label_col: str = "label_name",
    save_path: str | Path | None = None
):
    """
    Plot class distribution pie chart with count and percentage.
    """
    ...


def plot_image_size_distribution(
    audit_df: pd.DataFrame,
    save_path: str | Path | None = None
):
    """
    Plot width vs height distribution.
    """
    ...


def plot_rgb_channel_kde(
    df: pd.DataFrame,
    sample_per_class: int = 300,
    path_col: str = "path",
    label_col: str = "label_name",
    save_path: str | Path | None = None
):
    """
    Plot KDE of mean R/G/B channel intensity by class.
    """
    ...


def plot_metric_distribution(
    audit_df: pd.DataFrame,
    metric: str,
    label_col: str = "label_name",
    thresholds: list | None = None,
    save_path: str | Path | None = None
):
    """
    Plot metric distribution by class with optional threshold lines.
    """
    ...


def plot_metric_correlation_heatmap(
    audit_df: pd.DataFrame,
    metrics: list[str],
    save_path: str | Path | None = None
):
    """
    Plot correlation matrix of audit metrics.
    """
    ...


def plot_removed_examples(
    removed_df: pd.DataFrame,
    reason: str | None = None,
    n: int = 8,
    save_path: str | Path | None = None
):
    """
    Show removed images, optionally filtered by removal reason.
    """
    ...


def plot_before_after_cleaning(
    summary_df: pd.DataFrame,
    save_path: str | Path | None = None
):
    """
    Plot class counts before and after cleaning.
    """
    ...


def plot_transform_examples(
    image_paths: list[str],
    transform_dict: dict,
    save_path: str | Path | None = None
):
    """
    Show original image and transformed versions.
    """
    ...


def plot_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: str | Path | None = None
):
    """
    Plot Cat/Dog counts in train/val/test sets.
    """
    ...


def plot_grid_search_topk(
    result_df: pd.DataFrame,
    metric: str = "accuracy",
    top_k: int = 10,
    save_path: str | Path | None = None
):
    """
    Plot top-k configurations from grid search.
    """
    ...


def plot_grid_search_heatmap(
    result_df: pd.DataFrame,
    row: str = "preprocessing",
    col: str = "backbone",
    value: str = "accuracy",
    save_path: str | Path | None = None
):
    """
    Plot heatmap of best metric by preprocessing and backbone.
    """
    ...


def plot_confusion_matrix(
    cm,
    labels=("Cat", "Dog"),
    save_path: str | Path | None = None
):
    """
    Plot confusion matrix heatmap.
    """
    ...


def plot_wrong_predictions(
    wrong_df: pd.DataFrame,
    n: int = 8,
    save_path: str | Path | None = None
):
    """
    Plot wrong prediction cases with true/predicted labels.
    """
    ...


def plot_training_curves(
    history_df: pd.DataFrame,
    save_path_prefix: str | Path | None = None
):
    """
    Plot training/validation accuracy and loss curves for end-to-end DL.
    """
    ...


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "f1_macro",
    save_path: str | Path | None = None
):
    """
    Plot model comparison bar chart.
    Should only be used when more than one model is compared.
    """
    ...