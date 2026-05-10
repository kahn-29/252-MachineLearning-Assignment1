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

## Module API Reference (source-of-truth: modules/ code)

This API section was regenerated to match the actual exported functions and constants implemented in the `modules/` package sources. Use these lists as the canonical reference for notebook imports.

### `modules/__init__.py`

Lightweight package marker. Exposes submodules under the `modules` package and a `__version__` string.

- `__version__` — package version string
- package submodules: `artifacts`, `backbones`, `classical_models`, `cleaning`, `config_types`, `config_utils`, `data_utils`, `datasets`, `deep_learning`, `evaluation`, `feature_extraction`, `grid_search`, `image_audit`, `threshold_experiments`, `transforms`, `visualization`

### `modules/config_utils.py`

Configuration and runtime utilities. This module relies on types/constants defined in `modules.config_types`.

- `PROJECT_NAME`, `DEFAULT_SEED`, `SUPPORTED_PREPROCESSING_MODES`, `SUPPORTED_BACKBONES`, `SUPPORTED_CLASSIFIERS` (imported from `config_types`)
- `set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None`
- `get_device() -> Any`
- `ensure_dirs(*dirs) -> None`
- `resolve_workspace(workspace=None, project_name=PROJECT_NAME) -> Path`
- `get_default_config(user_config: Mapping[str, Any] | None = None) -> dict`
- `validate_config(config: Mapping[str, Any]) -> None`
- `config_to_run_name(config: Mapping[str, Any]) -> str`
- `save_config(config: Mapping[str, Any], path: str | Path) -> None`
- `load_config(path: str | Path) -> dict`
- `validate_and_normalize(name: str, supported_list: tuple[str, ...], entity_name: str = "option") -> str`

Note: `deep_update` is not implemented in the current source; use `FullConfig.from_dict(...)` (via `get_default_config`) for config merging/validation.

### `modules/data_utils.py`

Dataset discovery, dataframe creation, stratified splitting, and summaries.

- `DEFAULT_EXTENSIONS`
- `resolve_dataset_root(dataset_id=None, local_root=None, kaggle_input_dir="/kaggle/input", extensions=None, allow_download=True) -> Path`
- `list_image_paths(root, extensions=None) -> list[Path]`
- `infer_label_from_path(path, class_map=None) -> tuple[int|None, str|None]`
- `build_raw_dataframe(root, extensions=None, class_map=None, drop_unknown=True) -> pd.DataFrame`
- `stratified_split(df, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42, label_col="label") -> (train_df, val_df, test_df)`
- `sample_by_class(df, n_per_class=5, label_col="label_name", seed=42) -> pd.DataFrame`
- `summarize_class_distribution(df, label_col="label_name") -> pd.DataFrame`
- `summarize_split_distribution(train_df, val_df, test_df, label_col="label_name") -> pd.DataFrame`
- `summarize_splits_distribution(splits, label_col="label_name") -> pd.DataFrame`
- `read_image_metadata(path) -> dict`
- `add_image_metadata(df, path_col="path") -> pd.DataFrame`
- `remove_spatial_outliers(df, cols=("width","height"), n_std=3.0) -> pd.DataFrame`

Expected raw dataframe columns (from `build_raw_dataframe`): `sample_id`, `path`, `label`, `label_name`, `filename`, `extension`.

### `modules/image_audit.py`

Image-quality metric computation utilities.

- `DEFAULT_AUDIT_METRICS`
- `read_image_cv2(path) -> np.ndarray | None`
- `read_image_pil(path) -> PIL.Image | None`
- `compute_laplacian_variance(gray) -> float`
- `compute_entropy(gray) -> float`
- `compute_brightness_stats(gray) -> dict`
- `compute_gray_tolerance(rgb) -> dict`
- `compute_near_mono_metrics(gray) -> dict`
- `compute_aspect_metrics(width, height) -> dict`
- `compute_saturation_metrics(bgr) -> dict`
- `compute_chromaticity_metrics(rgb) -> dict`
- `compute_center_saliency(gray) -> dict`
- `compute_compression_artifact(gray) -> float`
- `compute_phash(path) -> str | None` (requires `imagehash`)
- `inspect_image(path, label=None, label_name=None, compute_hash=True) -> dict`
- `audit_dataframe(df, path_col="path", label_col="label", label_name_col="label_name", compute_hash=True, preserve_cols=None, show_progress=True) -> pd.DataFrame`
- `describe_audit_metrics(audit_df, metrics=None) -> pd.DataFrame`

Produced audit columns include size, brightness, blur, entropy, gray-tolerance metrics, saturation/chroma metrics, saliency, compression artifact, and `phash` when available.

### `modules/cleaning.py`

Image-cleaning decision helpers and duplicate detection.

- `compute_quality_score(audit_df) -> pd.DataFrame`
- `mark_near_duplicates(audit_df, hamming_threshold=4, hash_col="phash", bands=8) -> pd.DataFrame`
- `compute_soft_flags(audit_df, cleaning_config) -> pd.DataFrame`
- `build_cleaning_mask(audit_df, cleaning_config) -> pd.Series`
- `assign_removal_reasons(audit_df, cleaning_config) -> pd.DataFrame`
- `apply_cleaning(audit_df, cleaning_config, reset_index: bool = False) -> (clean_df, removed_df)`
- `summarize_cleaning(raw_df, clean_df, removed_df, label_col="label_name") -> pd.DataFrame`
- `summarize_removal_reasons(removed_df, reason_col=None) -> pd.DataFrame`
- `evaluate_cleaning_retention(raw_df, clean_df, label_col="label") -> dict`

Constants and internal helpers are used for flag names and quality-weighted ranking. Duplicate clustering uses a small union-find implementation.

### `modules/threshold_experiments.py`

Utilities to evaluate cleaning masks using a fixed proxy training protocol (logistic regression + scaling).

- `get_default_threshold_specs() -> list[dict]`
- `get_default_cleaning_presets() -> dict[str, dict]`
- `build_single_metric_mask(audit_df, metric, threshold, direction, base_mask=None) -> pd.Series`
- `make_proxy_split(audit_df, valid_mask, val_size, seed, label_col='label') -> (train_idx, val_idx)`
- `evaluate_cleaning_mask(mask, X_all, y_all, audit_df, train_indices, val_indices, mask_name='mask', threshold='threshold', seed=42, classifier_C=0.1, retention_penalty=0.10, imbalance_penalty=0.05, min_train_samples=100) -> dict`
- `run_single_metric_threshold_sweep(X, y, audit_df, threshold_specs, train_indices, val_indices, base_mask, proxy_config=None, score_config=None, seed=42) -> pd.DataFrame`
- `evaluate_cleaning_presets(X, y, audit_df, presets, train_indices, val_indices, proxy_config=None, score_config=None, seed=42) -> pd.DataFrame`
- `run_cleaning_stability_check(X, y, audit_df, presets, seeds, val_size, proxy_config=None, score_config=None) -> (stability_df, stability_summary)`
- `select_cleaning_policy(stability_summary, presets, min_delta_f1, min_retention_pct, max_imbalance_shift, default_policy) -> (selected_preset, selected_config, selected_row, selection_rule)`
- `build_cleaning_report_payload(selected_preset, selected_config, selected_row, selection_rule, audit_df, final_clean_df, removed_df, config) -> dict`

### `modules/transforms.py`

Image preprocessing transforms and convenience converters.

- `SUPPORTED_TRANSFORM_MODES` = ("stretch","center_crop","letterbox","augmented")
- `LetterBox(size, fill=(0,0,0), interpolation=Image.Resampling.LANCZOS)`
- `SquarePad(fill=(0,0,0))`
- `get_imagenet_mean_std() -> (mean, std)`
- `get_normalize_transform(normalize='imagenet') -> torchvision.transforms.Normalize | Lambda`
- `get_hybrid_transform(mode, image_size=224, train=False, normalize='imagenet') -> torchvision.transforms.Compose`
- `get_dl_transform(image_size=224, train=False, normalize='imagenet') -> torchvision.transforms.Compose`
- `build_image_transform(config: dict, split='train') -> torchvision.transforms.Compose`
- `tensor_to_display_image(tensor: torch.Tensor, denormalize=True, normalize='imagenet') -> np.ndarray`
- `transform_image_to_array(path, transform, dtype='float32') -> np.ndarray`

### `modules/datasets.py`

PyTorch Dataset and DataLoader utilities.

- `ImagePathDataset(df=None, paths=None, labels=None, transform=None, path_col='path', label_col='label', return_labels=True, fallback_size=224, on_error='raise')`
- `NpyBatchDataset(batch_dir, split_name, mmap_mode='r', return_tensors=True)`
- `create_image_dataloader(df, transform, batch_size, shuffle=False, num_workers=0, pin_memory=True, path_col='path', label_col='label', return_labels=True, on_error='raise') -> DataLoader`
- `create_image_dataloaders(splits, transforms, batch_size, num_workers=0, pin_memory=True, path_col='path', label_col='label', return_labels=True, on_error='raise') -> dict[str,DataLoader]`
- `create_npy_batch_dataloader(batch_dir, split_name, batch_size, shuffle=False, num_workers=0, pin_memory=True, mmap_mode='r', return_tensors=True) -> DataLoader`

### `modules/backbones.py`

Pretrained backbone loading and transfer-model utilities (wraps torchvision models).

- `get_backbone(name, device=None, pretrained=True, freeze=True, data_parallel=False) -> nn.Module`
- `get_feature_dim(backbone_name) -> int`
- `freeze_model(model) -> nn.Module`
- `unfreeze_model(model) -> nn.Module`
- `replace_classifier_head(model, backbone_name, num_classes, dropout=0.2) -> nn.Module`
- `build_transfer_model(backbone_name, num_classes, pretrained=True, dropout=0.2, freeze_backbone=True, device=None) -> nn.Module`

Notes:
- The module validates backbone names against `SUPPORTED_BACKBONES` (from `config_types`). Some torchvision backbones are implemented (`resnet18`, `vgg16`, `efficientnet_b0`) and their feature dims are defined.

### `modules/feature_extraction.py`

Frozen-backbone feature extraction and cache management.

- `extract_features(df, transform, backbone_name, batch_size=128, device=None, num_workers=0, path_col='path', label_col='label', pretrained=True, data_parallel=False, show_progress=True, on_error='raise') -> (X,y)`
- `extract_feature_splits(train_df, val_df, test_df, train_transform, eval_transform, backbone_name, batch_size=128, device=None, num_workers=0, output_dir=None, pretrained=True, data_parallel=False, force_recompute=False, path_col='path', label_col='label', show_progress=True, on_error='raise') -> dict`
- `load_or_extract_feature_splits(splits, transforms, config, output_dir=None) -> dict`

Caches use files `X_<split>.npy`, `y_<split>.npy` and a `feature_manifest.json` to record backbone and split sizes.

### `modules/classical_models.py`

Classical ML classifier construction and tuning helpers.

- `get_param_grid(classifier_name, grid_size='small') -> dict`
- `get_classifier(name, seed=42, params: dict | None = None, **overrides) -> sklearn.BaseEstimator`
- `train_classifier(X_train, y_train, classifier_name, params: dict | None = None, seed=42, **overrides) -> BaseEstimator`
- `tune_classifier_grid(X_train, y_train, classifier_name, param_grid=None, base_params=None, grid_size='small', cv=3, seed=42, scoring='f1_macro', refit=None, n_jobs=-1, verbose=1, return_train_score=False) -> GridSearchCV`

Supported classifier names in code include `logistic_regression`, `svm_linear`, `random_forest`, `voting_soft`, and `stacking`. The module builds pipelines/ensembles and supports staged tuning for ensembles.

### `modules/deep_learning.py`

End-to-end transfer-learning utilities.

- `unfreeze_last_blocks(model, num_blocks=1) -> nn.Module`
- `create_image_dataloaders_for_config(train_df, val_df, test_df, config) -> dict[str, DataLoader]`
- `build_model_for_config(config, num_classes, device=None) -> nn.Module`
- `build_optimizer(model, config) -> torch.optim.Optimizer`
- `build_scheduler(optimizer, config) -> Optional[scheduler]`
- `train_one_epoch(model, dataloader, criterion, optimizer, device) -> dict`
- `evaluate_one_epoch(model, dataloader, criterion, device) -> dict`
- `fit_transfer_model(model, dataloaders, config) -> dict` (returns model, history DataFrame, best_val_f1_macro)
- `predict_dataloader(model, dataloader, device) -> (y_true, y_pred, y_prob)`
- `save_checkpoint(model, optimizer, epoch, metrics, path) -> None`
- `load_checkpoint(path, model, optimizer=None, map_location=None) -> dict`
- `count_trainable_parameters(model) -> dict`

### `modules/evaluation.py`

Classification evaluation utilities.

- `compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict`
- `classification_report_df(y_true, y_pred, labels=None, label_names=None) -> pd.DataFrame`
- `confusion_matrix_df(y_true, y_pred, labels=None, label_names=None) -> pd.DataFrame`
- `evaluate_predictions(y_true, y_pred, y_prob=None, labels=None, label_names=None) -> dict`
- `evaluate_estimator(model, X, y, model_name='model') -> dict`
- `evaluate_model` (alias)
- `find_wrong_predictions(df, y_true, y_pred, y_prob=None, path_col='path', label_map=None) -> pd.DataFrame`
- `format_metrics_table(metrics_dict) -> pd.DataFrame`
- `compare_pipeline_results(results: Mapping[str, Mapping[str,Any]]) -> pd.DataFrame`

### `modules/grid_search.py`

Classical pipeline grid-search utilities.

- `generate_experiment_grid(search_space) -> list[dict]`
- `merge_experiment_config(base_config, experiment_config) -> dict`
- `run_single_classical_experiment(train_df, val_df, test_df, config, feature_dir, device=None, run_name=None) -> dict`
- `run_grid_search(train_df, val_df, test_df, base_config, search_space, output_dir, device=None, fail_fast=True) -> pd.DataFrame`
- `rank_grid_results(results_df, primary_metric='f1_macro', tie_breakers=None) -> pd.DataFrame`
- `select_default_config(ranked_results, base_config, config_columns=None) -> dict`
- `select_default_config_from_grid` (alias)

### `modules/artifacts.py`

Artifact saving and loading utilities.

- `save_json(data, path, indent=2) -> None`
- `load_json(path) -> dict`
- `save_dataframe(df, path, index=False) -> None`
- `load_dataframe(path) -> pd.DataFrame`
- `save_numpy(array, path) -> None`
- `load_numpy(path) -> np.ndarray`
- `save_pickle(obj, path) -> None`
- `save_feature_split(X, y, split_name, output_dir) -> dict`
- `load_feature_split(split_name, feature_dir) -> (X,y)`
- `feature_files_exist(feature_dir, split_names=('train','val','test')) -> bool`

Note: there is no `load_pickle` function implemented in the current `modules/artifacts.py` source; only `save_pickle` is present.

### `modules/visualization.py`

Plotting utilities used by notebooks.

- `plot_image_grid_from_df(df, n=12, path_col='path', title_col=None, subtitle_cols=None, filter_col=None, filter_value=None, sort_by=None, ascending=True, random_sample=False, seed=42, n_cols=4, figsize=None, suptitle=None, save_path=None, show=False) -> (Figure, axes)`
- `plot_sample_grid(df, n_per_class=5, path_col='path', label_col='label_name', class_order=None, seed=42, figsize=None, save_path=None, show=False) -> (Figure, axes)`
- `plot_pie_chart(df, category_col, title=None, save_path=None, show=False)`
- `plot_bar_chart(df, category_col, title=None, x_label=None, y_label='Count', save_path=None, show=False)`
- `plot_scatter_distribution(df, x_col, y_col, hue_col=None, title=None, save_path=None, show=False)`
- `plot_rgb_channel_kde(df, sample_per_class=300, path_col='path', label_col='label_name', class_order=None, seed=42, save_path=None, show=False)`
- `plot_metric_distribution(audit_df, metric, label_col='label_name', thresholds=None, title=None, save_path=None, show=False)`
- `plot_metric_correlation_heatmap(audit_df, metrics, title='Correlation Matrix of Image Quality Metrics', save_path=None, show=False)`
- `plot_before_after_cleaning(summary_df, label_col='label_name', save_path=None, show=False)`
- `plot_threshold_sweep_results(sweep_df, metric_col='f1_macro', save_path=None, show=False)`
- `plot_split_distribution(splits=None, train_df=None, val_df=None, test_df=None, label_col='label_name', save_path=None, show=False)`
- `plot_confusion_matrix(cm, labels=None, title='Confusion Matrix', save_path=None, show=False)`
- `plot_grid_search_results(results_df, x='feature_extraction.backbone', y='f1_macro', hue='classifier.name', save_path=None, show=False)`

## Public and Private API Rule

Public API functions (those intended for notebook use) are exported via the modules' `__all__` variables or documented above. Private helpers begin with a leading underscore `_` and are implementation details.

When using the package, prefer importing specific functions from the modules, e.g.:

```python
from modules.data_utils import build_raw_dataframe, stratified_split
from modules.feature_extraction import load_or_extract_feature_splits
```

## Notes about source/documentation parity

- This document is now aligned with the code currently present in the `modules/` folder. If you later modify or add functions in code, please regenerate this section or open a small docs PR to keep the reference accurate.

