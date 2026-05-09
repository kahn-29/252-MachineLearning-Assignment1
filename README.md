# Cat vs. Dog Image Classification Pipeline

**Course**: Machine Learning (C03117)
**Semester**: Semester I, Academic Year 2025–2026
**University**: Ho Chi Minh City University of Technology, VNU-HCM
**Department**: Faculty of Computer Science and Engineering
**Instructor**: Dr. Truong Vinh Lan
**Project Type**: Machine Learning Major Assignment
**Task**: Binary image classification — Cat vs. Dog

---

## Project Summary

This repository contains a machine learning pipeline for binary image classification on cat and dog images. The core pipeline uses pretrained CNN backbones as frozen feature extractors and trains classical machine learning classifiers on the extracted feature vectors. An optional end-to-end deep learning pipeline is included for comparison.

The pipeline consists of the following stages:

* Dataset download and discovery in Google Colab
* Exploratory data analysis (EDA)
* Image metadata analysis
* Image quality auditing
* Data cleaning and threshold analysis
* Image preprocessing and transformation
* Feature extraction using pretrained CNN backbones
* Feature caching as `.npy` files
* Classical classifier training
* Hyperparameter tuning
* Model evaluation
* Visualization and artifact export
* Optional transfer-learning pipeline

Execution environment: Google Colab only. Local setup and Kaggle runtime not required.

---

## Assignment Alignment

| Requirement                | Implementation                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| Google Colab execution     | Main workflow executed through Colab notebooks in `notebooks/`.                              |
| Traditional ML pipeline    | Frozen feature extraction with classical classifiers.                                        |
| Image EDA                  | Class distribution, image size, channel analysis, and RGB statistics.                        |
| Preprocessing              | Resize, crop, letterbox, and augmentation modes.                                             |
| Feature extraction         | Pretrained CNN backbones: VGG16, ResNet18, ResNet50, EfficientNet-B0, EfficientNet-B2.       |
| Feature storage            | Features saved as `.npy` files with manifests.                                               |
| Classifier training        | Logistic Regression, Linear SVM, Random Forest, Soft Voting, Stacking.                       |
| Evaluation metrics         | Accuracy, precision, recall, F1-score, confusion matrix, wrong predictions.                  |
| Experiment support         | Grid search across preprocessing, backbone, and classifier variations.                      |
| Transfer learning          | Optional end-to-end transfer-learning pipeline.                                              |
| Output structure           | `notebooks/`, `modules/`, `features/`, `reports/`, `results/`, `models/` directories.        |

---

## Repository Contents

```text
cat-dog-image-classifier/
├── README.md
├── MODULES.md
├── notebooks/
│   ├── classical_pipeline.ipynb
│   ├── cleaning_threshold_experiments.ipynb
│   └── deep_learning_pipeline.ipynb
├── modules/
│   ├── __init__.py
│   ├── config_utils.py
│   ├── data_utils.py
│   ├── image_audit.py
│   ├── cleaning.py
│   ├── threshold_experiments.py
│   ├── transforms.py
│   ├── datasets.py
│   ├── backbones.py
│   ├── feature_extraction.py
│   ├── classical_models.py
│   ├── deep_learning.py
│   ├── evaluation.py
│   ├── grid_search.py
│   ├── artifacts.py
│   └── visualization.py
├── features/
├── results/
├── reports/
│   └── figures/
├── models/
└── data/
```

Generated folders such as `data/`, `features/`, `results/`, `models/`, and `reports/figures/` may be empty in the repository. The notebooks regenerate these artifacts when executed on Colab.

This repository does **not** use a `requirements.txt` file. Required libraries are installed directly inside the notebook setup cells.

---

## File and Folder Map

### Root files

* `README.md`: repository overview, workflow, Colab run instructions, outputs, and project notes.
* `MODULES.md`: detailed package architecture and module API reference.

### `notebooks/`

* `classical_pipeline.ipynb`: main classical machine learning workflow from dataset loading to final evaluation.
* `cleaning_threshold_experiments.ipynb`: image-quality threshold analysis and cleaning-policy selection.
* `deep_learning_pipeline.ipynb`: optional transfer-learning workflow for comparison and bonus evaluation.

### `modules/`

* `__init__.py`: package marker.
* `config_utils.py`: config creation, validation, runtime setup, seed control, and path resolution.
* `data_utils.py`: dataset discovery, dataframe creation, label inference, splitting, and summaries.
* `image_audit.py`: image-quality metric computation.
* `cleaning.py`: cleaning decisions based on audit metrics and config thresholds.
* `threshold_experiments.py`: cleaning-threshold sweeps and policy selection.
* `transforms.py`: preprocessing and augmentation transforms.
* `datasets.py`: PyTorch dataset and dataloader wrappers.
* `backbones.py`: pretrained backbone loading and transfer-model construction.
* `feature_extraction.py`: frozen deep-feature extraction and feature-cache handling.
* `classical_models.py`: classical classifier construction, training, and tuning.
* `deep_learning.py`: transfer-learning training, prediction, and checkpoint handling.
* `evaluation.py`: metrics, reports, confusion matrices, and wrong-prediction analysis.
* `grid_search.py`: classical experiment orchestration across preprocessing, backbone, and classifier settings.
* `artifacts.py`: saving and loading JSON, dataframes, NumPy arrays, feature splits, and pickle artifacts.
* `visualization.py`: plotting utilities for EDA, cleaning, evaluation, and reports.

### Output folders

* `data/`: dataset files downloaded or extracted during Colab execution.
* `features/`: cached feature arrays and feature manifests.
* `results/`: metrics, grid-search results, and cleaning reports.
* `reports/`: final report and generated figures.
* `models/`: saved trained models when included.

---

## Main Pipeline - Classical Machine Learning Pipeline

The classical pipeline is the required core pipeline of the project.

```text
Raw images
→ Dataset download/discovery in Colab
→ EDA and metadata analysis
→ Image audit
→ Cleaning policy selection
→ Train/validation/test split
→ Image preprocessing
→ Frozen CNN feature extraction
→ Save features as .npy
→ Train classical classifiers
→ Hyperparameter tuning
→ Validate and test models
→ Save metrics, figures, and models
```

This pipeline uses pretrained CNNs as fixed feature extractors. The CNN backbone produces feature vectors, and classical machine learning models are trained on top of those vectors.

Supported backbones:

* VGG16
* ResNet18
* ResNet50
* EfficientNet-B0
* EfficientNet-B2

Supported classifiers:

* Logistic Regression
* Linear SVM
* Random Forest
* Soft Voting Classifier
* Stacking Classifier

---

## Optional Deep Learning Pipeline

The deep learning pipeline is an optional extension for comparison.

```text
Raw images
→ Dataset download/discovery in Colab
→ Cleaning and split reuse
→ Deep learning transforms and augmentation
→ Transfer-learning model construction
→ Head training
→ Optional fine-tuning
→ Validation monitoring
→ Test evaluation
→ Comparison with classical pipeline
```

This pipeline replaces the final classification head of a pretrained backbone and trains it for the cat/dog classification task.

---

## Dataset

The recommended dataset source is:

```text
tongpython/cat-and-dog
```

The dataset is downloaded or prepared directly inside the Colab notebook. The final notebooks should not depend on personal cloud storage such as Google Drive, Dropbox, or OneDrive.

---

## Colab Execution

Open `notebooks/1_Traditional_Machine_Learning_Pipeline.ipynb` and select `Runtime → Run all`.

Setup cells in the notebook perform:

1. Library installation
2. Repository access
3. Module path configuration
4. Dataset download or discovery
5. Output directory creation
6. Pipeline execution


---

## Dependencies

Most core libraries are already available in Colab. The notebook installs any missing libraries directly when executing.

---

## Execution Workflow

### Classical Machine Learning Pipeline

Notebook: `notebooks/1_Traditional_Machine_Learning_Pipeline.ipynb`

Pipeline stages:

1. Environment setup
2. Module import
3. Dataset loading
4. Exploratory data analysis
5. Image quality audit
6. Data cleaning
7. Train/validation/test split
8. Preprocessing transform creation
9. Feature extraction
10. Feature caching
11. Classifier training
12. Hyperparameter tuning
13. Validation and test evaluation
14. Metrics and visualization
15. Artifact export

---

### Cleaning Threshold Analysis

Notebook: `notebooks/experiments/Sweet_Spot_Threshold.ipynb`

Analyzes cleaning thresholds and selects optimal cleaning parameters.

---

### Deep Learning Pipeline

Notebook: `notebooks/2_Deep_Learning_Pipeline.ipynb`

Trains end-to-end transfer-learning model with comparison to classical pipeline.

Outputs:

```text
models/dl_best_<backbone>.pt
results/deep_learning_metrics.json
reports/figures/dl_training_curves.png
reports/figures/dl_confusion_matrix.png
```

---

## Configuration

Pipeline behavior is controlled by a config dictionary.

Default configuration:

```python
config = {
    "project": {
        "name": "cat-dog-image-classifier",
        "task": "binary_image_classification",
        "target": "cat_vs_dog",
    },
    "seed": 42,
    "runtime": {
        "device": "auto",
        "num_workers": 2,
        "deterministic": False,
    },
    "dataset": {
        "kaggle_id": "tongpython/cat-and-dog",
        "local_root": None,
    },
    "split": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
        "seed": 42,
    },
    "cleaning": {
        "remove_corrupted": True,
        "remove_duplicates": False,
        "duplicate_hamming_threshold": 4,
        "min_side": 64,
        "max_aspect_extremity": 5.0,
        "min_blur_laplacian": 40.0,
        "max_near_mono_ratio": 0.92,
        "max_dark_ratio": 0.98,
        "max_bright_ratio": 0.98,
    },
    "preprocessing": {
        "mode": "augmented",
        "image_size": 224,
        "train_augmentation": False,
        "normalize": "imagenet",
    },
    "feature_extraction": {
        "backbone": "efficientnet_b0",
        "batch_size": 64,
        "num_workers": 2,
        "pretrained": True,
        "data_parallel": False,
    },
    "classifier": {
        "name": "logistic_regression",
        "params": {},
    },
    "hyperparameter_tuning": {
        "enabled": True,
        "grid_size": "small",
        "cv": 3,
        "scoring": "f1_macro",
        "n_jobs": -1,
    },
}
```

Configuration validation:

```python
from modules.config_utils import get_default_config, validate_config

config = get_default_config()
validate_config(config)
```

---

## Feature Caching

Extracted features are saved as `.npy` files.

Folder structure:

```text
features/
└── efficientnet_b0_224_augmented/
    ├── X_train.npy
    ├── y_train.npy
    ├── X_val.npy
    ├── y_val.npy
    ├── X_test.npy
    ├── y_test.npy
    └── manifest.json
```

Manifest contents:

```text
backbone_name
pretrained
image_size
preprocessing_mode
feature_dim
split_sizes
created_at
```

---

## Experiments

Example search space:

```python
search_space = {
    "preprocessing.mode": ["stretch", "center_crop", "letterbox", "augmented"],
    "preprocessing.image_size": [224],
    "feature_extraction.backbone": ["vgg16", "resnet18", "efficientnet_b0"],
    "classifier.name": [
        "logistic_regression",
        "svm_linear",
        "random_forest",
        "voting_soft",
        "stacking",
    ],
}
```

Example experiment scale:

```text
4 preprocessing modes × 1 image size × 3 backbones × 11 classifier variants = 132 experiments
```

Primary ranking metric: macro F1-score

Secondary metrics: validation accuracy, inference time, training time

---

## Evaluation Metrics

The project reports:

| Metric            | Meaning                                           |
| ----------------- | ------------------------------------------------- |
| Accuracy          | Overall percentage of correct predictions.        |
| Precision         | Fraction of predicted positives that are correct. |
| Recall            | Fraction of actual positives that are detected.   |
| F1-score          | Harmonic mean of precision and recall.            |
| Macro F1-score    | Average F1-score across classes.                  |
| Confusion Matrix  | Correct and incorrect predictions by class.       |
| Error Rate        | Percentage of incorrect predictions.              |
| Wrong Predictions | Misclassified images for qualitative review.      |

The main comparison includes validation accuracy, validation macro F1-score, test accuracy, test macro F1-score, training time, inference time, wrong-prediction count, and confusion matrix.

---

## Reproducibility Requirements

* Fixed random seed throughout pipeline
* Deterministic dataset splitting with seed saving
* Configuration JSON saved for each experiment run
* Feature cache manifests for tracking preprocessing parameters
* Incremental grid-search result logging
* Public dataset links in notebooks
* Support for `Runtime → Run all` execution
* No dependencies on private cloud storage or local paths

---

## License and Usage

This repository is created for academic purposes as part of the Machine Learning course major assignment.

The code may be reused for learning, experimentation, and educational demonstrations. Dataset usage should follow the license and terms of the original dataset provider.

---

## Libraries and Dependencies

Core libraries: PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib, OpenCV, Pillow

---

**Course**: Machine Learning (C03117), HCMUT
**Last Updated**: May 2026
