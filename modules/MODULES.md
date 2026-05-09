# MODULES.md — Package Architecture & API Reference

**Project**: Cat vs. Dog Image Classification Pipeline  
**Course**: Machine Learning (C03117)  
**Version**: 1.1.0  
**Last Updated**: May 2026

This document defines the final structure of the `modules/` package for the cat/dog binary image-classification project. The `notebooks/` provide orchestration, explanation, execution, and visualization. The `modules/` folder provides reusable Python backend logic.

## Design Principles

- Each module has one narrow responsibility.
- Major pipeline choices are controlled by a shared config dictionary.
- Seeds, deterministic behavior, cache manifests, and saved configs support reproducibility.
- Dataset handling, auditing, cleaning, preprocessing, feature extraction, modeling, evaluation, and visualization are separated.
- Notebook code stays thin and imports reusable logic from `modules/`.
- The code is usable in local environments, Colab, and Kaggle.
- Public functions have docstrings, type hints where practical, and clear inputs and outputs.

## Final Folder Structure

```text
modules/
├── __init__.py
├── config_utils.py
├── data_utils.py
├── image_audit.py
├── cleaning.py
├── threshold_experiments.py
├── transforms.py
├── datasets.py
├── backbones.py
├── feature_extraction.py
├── classical_models.py
├── deep_learning.py
├── evaluation.py
├── grid_search.py
├── artifacts.py
└── visualization.py
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `config_utils.py` | Configuration, runtime setup, seed control, path resolution, config validation. |
| `data_utils.py` | Dataset discovery, dataframe construction, label inference, splitting, class summaries, metadata reading. |
| `image_audit.py` | Image-quality metric computation only. |
| `cleaning.py` | Cleaning decisions from audit metrics and cleaning config. |
| `threshold_experiments.py` | Threshold experiments for selecting cleaning policies. |
| `transforms.py` | Image preprocessing and augmentation transforms only. |
| `datasets.py` | PyTorch Dataset and DataLoader abstractions only. |
| `backbones.py` | Pretrained backbone loading and transfer-learning model construction. |
| `feature_extraction.py` | Frozen deep-feature extraction and feature-cache handling. |
| `classical_models.py` | Classical ML classifier construction, training, and tuning. |
| `deep_learning.py` | End-to-end transfer-learning training and inference. |
| `evaluation.py` | Metrics, reports, confusion matrices, wrong-prediction analysis. |
| `grid_search.py` | Classical-pipeline experiment orchestration over preprocessing, backbone, and classifier configs. |
| `artifacts.py` | Saving and loading JSON, dataframes, NumPy arrays, features, models, and metadata. |
| `visualization.py` | Plotting functions only. |

## Project Workflow

### Classical Machine Learning Pipeline

```text
Raw images
→ Dataset discovery
→ EDA and image metadata analysis
→ Image audit
→ Cleaning policy / threshold experiment
→ Train/validation/test split
→ Image preprocessing transform
→ Frozen pretrained backbone feature extraction
→ Save features as .npy
→ Train classical classifiers
→ Hyperparameter tuning
→ Evaluate on validation/test sets
→ Save reports, models, figures, and metrics
```

### End-to-End Deep Learning Pipeline

```text
Raw images
→ Dataset discovery
→ Cleaning and split reuse
→ DL transforms and augmentation
→ Transfer-learning model construction
→ Head training / optional fine-tuning
→ Validation monitoring
→ Test evaluation
→ Compare against classical pipeline
```

## Configuration Conventions

The project uses one nested config dictionary. Stable top-level sections are:

```python
config = {
    "project": {...},
    "seed": 42,
    "runtime": {...},
    "paths": {...},
    "dataset": {...},
    "split": {...},
    "cleaning": {...},
    "preprocessing": {...},
    "feature_extraction": {...},
    "classifier": {...},
    "hyperparameter_tuning": {...},
    "deep_learning": {...},
    "grid_search": {...},
}
```

 Snake_case keys are used throughout.

## Module API Reference

### `modules/__init__.py`

Package marker for the `modules` folder.

- `__version__`: package version string.
- `__all__`: public submodule list.

The file stays lightweight. It does not import heavy libraries or re-export many functions.

### `modules/config_utils.py`

Configuration and runtime utilities.

- `PROJECT_NAME`
- `DEFAULT_SEED`
- `SUPPORTED_PREPROCESSING_MODES`
- `SUPPORTED_BACKBONES`
- `SUPPORTED_CLASSIFIERS`
- `set_seed(seed=42, deterministic=False)`
- `get_device()`
- `ensure_dirs(*dirs)`
- `resolve_workspace(workspace=None, project_name=PROJECT_NAME)`
- `deep_update(default, override=None)`
- `get_default_config()`
- `validate_config(config)`
- `config_to_run_name(config)`
- `save_config(config, path)`
- `load_config(path)`

### `modules/data_utils.py`

Dataset discovery, dataframe creation, stratified splitting, and dataset summaries.

- `DEFAULT_EXTENSIONS`
- `resolve_dataset_root(dataset_id=None, local_root=None, kaggle_input_dir="/kaggle/input", extensions=None, allow_download=True)`
- `list_image_paths(root, extensions=None)`
- `infer_label_from_path(path, class_map=None)`
- `build_raw_dataframe(root, extensions=None, class_map=None, drop_unknown=True)`
- `stratified_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, label_col="label")`
- `sample_by_class(df, n_per_class=5, label_col="label_name", seed=42)`
- `summarize_class_distribution(df, label_col="label_name")`
- `summarize_split_distribution(train_df, val_df, test_df, label_col="label_name")`
- `summarize_splits_distribution(splits, label_col="label_name")`
- `read_image_metadata(path)`
- `add_image_metadata(df, path_col="path")`
- `remove_spatial_outliers(df, cols=("width", "height"), n_std=3.0)`

Expected raw dataframe columns include `sample_id`, `path`, `label`, `label_name`, `filename`, and `extension`.

### `modules/image_audit.py`

Image-quality metric computation.

- `DEFAULT_AUDIT_METRICS`
- `read_image_cv2(path)`
- `read_image_pil(path)`
- `compute_laplacian_variance(gray)`
- `compute_entropy(gray)`
- `compute_brightness_stats(gray)`
- `compute_gray_tolerance(rgb)`
- `compute_near_mono_metrics(gray)`
- `compute_aspect_metrics(width, height)`
- `compute_saturation_metrics(bgr)`
- `compute_chromaticity_metrics(rgb)`
- `compute_center_saliency(gray)`
- `compute_compression_artifact(gray)`
- `compute_phash(path)`
- `inspect_image(path, label=None, label_name=None, compute_hash=True)`
- `audit_dataframe(df, path_col="path", label_col="label", label_name_col="label_name", compute_hash=True, preserve_cols=None, show_progress=True)`
- `describe_audit_metrics(audit_df, metrics=None)`

Expected audit columns include `path`, `label`, `label_name`, `readable`, `width`, `height`, `channels`, `aspect_ratio`, `aspect_extremity`, `min_side`, `laplacian_var`, `entropy`, `brightness_mean`, `brightness_std`, `near_mono_ratio`, `dark_ratio`, `bright_ratio`, `saturation_mean`, `saturation_std`, `chroma_mean`, `center_saliency_ratio`, `compression_artifact`, and `phash`.

### `modules/cleaning.py`

Image-cleaning decisions from audit metrics.

- `compute_quality_score(audit_df)`
- `mark_near_duplicates(audit_df, hamming_threshold=4, hash_col="phash", bands=8)`
- `compute_soft_flags(audit_df, cleaning_config)`
- `build_cleaning_mask(audit_df, cleaning_config)`
- `assign_removal_reasons(audit_df, cleaning_config)`
- `apply_cleaning(audit_df, cleaning_config)`
- `summarize_cleaning(raw_df, clean_df, removed_df, label_col="label_name")`
- `summarize_removal_reasons(removed_df, reason_col="removal_reasons")`
- `evaluate_cleaning_retention(raw_df, clean_df, label_col="label")`

This module uses only columns produced by `image_audit.py`. Thresholds come from `config["cleaning"]`.

### `modules/threshold_experiments.py`

Threshold experiment utilities for cleaning-policy selection.

- `get_default_threshold_specs()`
- `get_default_cleaning_presets()`
- `build_single_metric_mask(audit_df, metric, threshold, direction, base_mask=None)`
- `make_proxy_split(audit_df, valid_mask, val_size, seed, label_col="label")`
- `evaluate_cleaning_mask(mask, X_all, y_all, audit_df, train_indices, val_indices, mask_name="mask", threshold="threshold", seed=42, classifier_C=0.1, retention_penalty=0.10, imbalance_penalty=0.05, min_train_samples=100)`
- `run_single_metric_threshold_sweep(X, y, audit_df, threshold_specs, train_indices, val_indices, base_mask, proxy_config=None, score_config=None, seed=42)`
- `evaluate_cleaning_presets(X, y, audit_df, presets, train_indices, val_indices, proxy_config=None, score_config=None, seed=42)`
- `run_cleaning_stability_check(X, y, audit_df, presets, seeds, val_size, proxy_config=None, score_config=None)`
- `select_cleaning_policy(stability_summary, presets, min_delta_f1, min_retention_pct, max_imbalance_shift, default_policy)`
- `build_cleaning_report_payload(selected_preset, selected_config, selected_row, selection_rule, audit_df, final_clean_df, removed_df, config)`

### `modules/transforms.py`

Image preprocessing transforms.

- `SUPPORTED_TRANSFORM_MODES`
- `LetterBox(size, fill=(0, 0, 0), interpolation=Image.Resampling.LANCZOS)`
- `SquarePad(fill=(0, 0, 0))`
- `get_imagenet_mean_std()`
- `get_normalize_transform(normalize="imagenet")`
- `get_hybrid_transform(mode, image_size=224, train=False, normalize="imagenet")`
- `get_dl_transform(image_size=224, train=False, normalize="imagenet")`
- `build_image_transform(config, split="train")`
- `tensor_to_display_image(tensor, denormalize=True, normalize="imagenet")`
- `transform_image_to_array(path, transform, dtype="float32")`

Supported modes include `stretch`, `center_crop`, `letterbox`, and `augmented`.

### `modules/datasets.py`

PyTorch Dataset and DataLoader utilities.

- `ImagePathDataset(df=None, paths=None, labels=None, transform=None, path_col="path", label_col="label", return_labels=True, fallback_size=224, on_error="raise")`
- `NpyBatchDataset(batch_dir, split_name, mmap_mode="r", return_tensors=True)`
- `create_image_dataloader(df, transform, batch_size, shuffle=False, num_workers=0, pin_memory=True, path_col="path", label_col="label", return_labels=True, on_error="raise")`
- `create_image_dataloaders(splits, transforms, batch_size, num_workers=0, pin_memory=True, path_col="path", label_col="label", return_labels=True, on_error="raise")`
- `create_npy_batch_dataloader(batch_dir, split_name, batch_size, shuffle=False, num_workers=0, pin_memory=True, mmap_mode="r", return_tensors=True)`

`ImagePathDataset` converts images to RGB and exposes clear error behavior through `on_error`.

### `modules/backbones.py`

Pretrained backbone and transfer-model construction.

- `SUPPORTED_BACKBONES`
- `list_supported_backbones()`
- `get_backbone(name, device=None, pretrained=True, freeze=True, data_parallel=False)`
- `get_feature_dim(backbone_name)`
- `freeze_model(model)`
- `unfreeze_model(model)`
- `replace_classifier_head(model, backbone_name, num_classes, dropout=0.2)`
- `build_transfer_model(backbone_name, num_classes, pretrained=True, dropout=0.2, freeze_backbone=True, device=None)`

Recommended backbone names include `vgg16`, `resnet18`, `resnet50`, `efficientnet_b0`, and `efficientnet_b2`.

### `modules/feature_extraction.py`

Frozen-backbone feature extraction.

- `extract_features(df, transform, backbone_name, batch_size=128, device=None, num_workers=0, path_col="path", label_col="label", pretrained=True, data_parallel=False, show_progress=True, on_error="raise")`
- `extract_feature_splits(train_df, val_df, test_df, train_transform, eval_transform, backbone_name, batch_size=128, device=None, num_workers=0, output_dir=None, pretrained=True, data_parallel=False, force_recompute=False, path_col="path", label_col="label", show_progress=True, on_error="raise")`
- `load_or_extract_feature_splits(splits, transforms, config, output_dir=None)`

Feature caches include `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, and `manifest.json`.

### `modules/classical_models.py`

Classical ML classifier construction, training, benchmarking, and tuning.

- `SUPPORTED_CLASSIFIERS`
- `list_supported_classifiers()`
- `get_classifier(name, seed=42, **params)`
- `get_param_grid(classifier_name, grid_size="small")`
- `train_classifier(X_train, y_train, classifier_name, seed=42, **params)`
- `tune_classifier_grid(X_train, y_train, classifier_name, param_grid=None, cv=3, seed=42, scoring="f1_macro", n_jobs=-1, grid_size="small")`

Recommended classifier names include `logistic_regression`, `svm_linear`, `random_forest`, `voting_soft`, and `stacking`.

### `modules/deep_learning.py`

End-to-end transfer-learning utilities.

- `unfreeze_last_blocks(model, num_blocks=1)`
- `create_image_dataloaders_for_config(train_df, val_df, test_df, config)`
- `build_model_for_config(config, num_classes, device=None)`
- `build_optimizer(model, config)`
- `build_scheduler(optimizer, config)`
- `train_one_epoch(model, dataloader, criterion, optimizer, device)`
- `evaluate_one_epoch(model, dataloader, criterion, device)`
- `fit_transfer_model(model, dataloaders, config)`
- `predict_dataloader(model, dataloader, device)`
- `save_checkpoint(model, optimizer, epoch, metrics, path)`
- `load_checkpoint(path, model, optimizer=None, map_location=None)`
- `count_trainable_parameters(model)`

### `modules/evaluation.py`

Classification evaluation utilities.

- `compute_classification_metrics(y_true, y_pred, y_prob=None)`
- `classification_report_df(y_true, y_pred, labels=None, label_names=None)`
- `confusion_matrix_df(y_true, y_pred, labels=None, label_names=None)`
- `evaluate_predictions(y_true, y_pred, y_prob=None, labels=None, label_names=None)`
- `evaluate_estimator(model, X, y, model_name="model")`
- `evaluate_model`
- `find_wrong_predictions(df, y_true, y_pred, y_prob=None, path_col="path", label_map=None)`
- `format_metrics_table(metrics_dict)`
- `compare_pipeline_results(results)`

Minimum reported metrics include `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `error_rate`, `n_samples`, and `n_errors`. Binary ROC-AUC is included when probabilities are available.

### `modules/grid_search.py`

Classical pipeline grid-search utilities.

- `generate_experiment_grid(search_space)`
- `merge_experiment_config(base_config, experiment_config)`
- `run_single_classical_experiment(train_df, val_df, test_df, config, feature_dir, device=None, run_name=None)`
- `run_grid_search(train_df, val_df, test_df, base_config, search_space, output_dir, device=None, fail_fast=True)`
- `rank_grid_results(results_df, primary_metric="f1_macro", tie_breakers=None)`
- `select_default_config(ranked_results, base_config, config_columns=None)`
- `select_default_config_from_grid`

The search space usually covers `preprocessing.mode`, `preprocessing.image_size`, `feature_extraction.backbone`, and `classifier.name`.

### `modules/artifacts.py`

Artifact saving and loading utilities.

- `save_json(data, path, indent=2)`
- `load_json(path)`
- `save_dataframe(df, path, index=False)`
- `load_dataframe(path)`
- `save_numpy(array, path)`
- `load_numpy(path)`
- `save_pickle(obj, path)`
- `load_pickle(path)`
- `save_feature_split(X, y, split_name, output_dir)`
- `load_feature_split(split_name, feature_dir)`
- `feature_files_exist(feature_dir, split_names=("train", "val", "test"))`

JSON is used for configs, metrics, manifests, and small structured metadata. NPY is used for feature matrices and label arrays. Pickle is used only for trusted fitted objects.

### `modules/visualization.py`

Plotting-only utilities.

- `plot_image_grid_from_df(df, n=12, path_col="path", title_col=None, subtitle_cols=None, filter_col=None, filter_value=None, sort_by=None, ascending=True, random_sample=False, seed=42, n_cols=4, figsize=None, suptitle=None, save_path=None, show=False)`
- `plot_sample_grid(df, n_per_class=5, path_col="path", label_col="label_name", class_order=None, seed=42, figsize=None, save_path=None, show=False)`
- `plot_pie_chart(df, category_col, title=None, save_path=None, show=False)`
- `plot_bar_chart(df, category_col, title=None, xlabel=None, ylabel="Count", save_path=None, show=False)`
- `plot_scatter_distribution(df, x_col, y_col, hue_col=None, title=None, save_path=None, show=False)`
- `plot_rgb_channel_kde(df, sample_per_class=300, path_col="path", label_col="label_name", class_order=None, seed=42, save_path=None, show=False)`
- `plot_metric_distribution(audit_df, metric, label_col="label_name", thresholds=None, title=None, save_path=None, show=False)`
- `plot_metric_correlation_heatmap(audit_df, metrics, title="Correlation Matrix of Image Quality Metrics", save_path=None, show=False)`
- `plot_before_after_cleaning(summary_df, label_col="label_name", save_path=None, show=False)`
- `plot_threshold_sweep_results(sweep_df, metric_col="f1_macro", save_path=None, show=False)`
- `plot_split_distribution(splits=None, train_df=None, val_df=None, test_df=None, label_col="label_name", save_path=None, show=False)`
- `plot_transform_examples(transform_records, n_images=None, save_path=None, show=False)`
- `plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path=None, show=False)`
- `plot_grid_search_results(results_df, x="feature_extraction.backbone", y="f1_macro", hue="classifier.name", save_path=None, show=False)`

Plotting functions accept `save_path` and `show`. Figures used in reports are saved under `reports/figures/`.

## Public and Private API Rule

Public functions are imported into notebooks. Private helpers begin with `_` and stay internal to their module.

Example:

```python
from modules.data_utils import build_raw_dataframe, stratified_split
from modules.feature_extraction import load_or_extract_feature_splits
```

## Usage Examples

### Build Dataset and Splits

```python
from modules.config_utils import get_default_config, set_seed, validate_config
from modules.data_utils import resolve_dataset_root, build_raw_dataframe, stratified_split

config = get_default_config()
validate_config(config)
set_seed(config["seed"])

root = resolve_dataset_root(
    dataset_id=config["dataset"].get("kaggle_id"),
    local_root=config["dataset"].get("local_root"),
)

raw_df = build_raw_dataframe(root)
train_df, val_df, test_df = stratified_split(
    raw_df,
    train_ratio=config["split"]["train"],
    val_ratio=config["split"]["val"],
    test_ratio=config["split"]["test"],
    seed=config["split"]["seed"],
)
```

### Audit and Clean Images

```python
from modules.image_audit import audit_dataframe, describe_audit_metrics
from modules.cleaning import apply_cleaning, summarize_cleaning

audit_df = audit_dataframe(raw_df, show_progress=True)
metric_summary = describe_audit_metrics(audit_df)
clean_df, removed_df = apply_cleaning(audit_df, config["cleaning"])
cleaning_summary = summarize_cleaning(raw_df, clean_df, removed_df)
```

### Extract Frozen Deep Features

```python
from modules.transforms import build_image_transform
from modules.feature_extraction import load_or_extract_feature_splits

train_transform = build_image_transform(config["preprocessing"], split="train")
eval_transform = build_image_transform(config["preprocessing"], split="eval")

features = load_or_extract_feature_splits(
    splits={"train": train_df, "val": val_df, "test": test_df},
    transforms={"train": train_transform, "val": eval_transform, "test": eval_transform},
    config=config,
)

X_train, y_train = features["train"]
X_val, y_val = features["val"]
X_test, y_test = features["test"]
```

### Train and Evaluate Classical Classifier

```python
from modules.classical_models import train_classifier
from modules.evaluation import evaluate_estimator

model = train_classifier(
    X_train,
    y_train,
    classifier_name=config["classifier"]["name"],
    seed=config["seed"],
    **config["classifier"].get("params", {}),
)

val_result = evaluate_estimator(model, X_val, y_val, model_name=config["classifier"]["name"])
test_result = evaluate_estimator(model, X_test, y_test, model_name=config["classifier"]["name"])
```

### Run Classical Grid Search

```python
from modules.grid_search import run_grid_search, rank_grid_results

search_space = {
    "preprocessing.mode": ["stretch", "center_crop", "letterbox", "augmented"],
    "preprocessing.image_size": [224],
    "feature_extraction.backbone": ["vgg16", "resnet18", "efficientnet_b0"],
    "classifier.name": ["logistic_regression", "svm_linear", "random_forest", "voting_soft", "stacking"],
}

results_df = run_grid_search(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    base_config=config,
    search_space=search_space,
    output_dir=config["paths"]["results_dir"],
)

ranked_df = rank_grid_results(results_df, primary_metric="f1_macro")
```
