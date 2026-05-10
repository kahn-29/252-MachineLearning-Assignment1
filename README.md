# Cat vs. Dog Image Classification Pipeline

Machine Learning (C03117) major assignment for binary image classification on the Kaggle `tongpython/cat-and-dog` dataset.

This repository compares two pipelines:

1. **Hybrid classical pipeline** — frozen pretrained CNN feature extraction followed by traditional classifiers.
2. **Deep learning pipeline** — end-to-end transfer learning with a trainable classification head and fine-tuning.

The main reported operational model is **EfficientNet-B0 frozen features + Logistic Regression**. The deep learning notebook is included as an additional comparison pipeline.

## Project Information

| Item | Details |
|---|---|
| Course | Machine Learning (C03117) |
| Semester | Semester 252 |
| Academic Year | 2025 |
| University | Ho Chi Minh City University of Technology, VNU-HCM |
| Faculty | Faculty of Computer Science and Engineering |
| Instructor | Dr. Truong Vinh Lan |
| Task | Binary image classification: Cat vs. Dog |

## Team

- Nguyễn Mạnh Quốc Khánh — 2352525 — `khanh.nguyenmanh@hcmut.edu.vn`
- Phan Ngọc Lan Chi — 2352137 — `chi.phanlanchi1906@hcmut.edu.vn`
- Ngô Diễm Quyên — 2353031 — `quyen.ngo6905@hcmut.edu.vn`
- Trần Lâm Anh — 2352067 — `anh.tranlam@hcmut.edu.vn`
- Vũ Đức Việt Anh — 2352074 — `anh.vu3577@hcmut.edu.vn`

## Quick Start

### Run in Google Colab

1. Open `notebooks/1_Traditional_Machine_Learning_Pipeline.ipynb`.
2. Select **Runtime → Run all**.
3. Run `notebooks/2_Deep_Learning_Pipeline.ipynb` only when the deep learning comparison is needed.

The notebooks handle environment setup, dataset discovery/download, output-directory creation, model training, evaluation, and artifact export.

### Main notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/1_Traditional_Machine_Learning_Pipeline.ipynb` | Main required classical ML workflow |
| `notebooks/2_Deep_Learning_Pipeline.ipynb` | End-to-end transfer learning comparison |
| `notebooks/experiments/Grid_Search.ipynb` | Exhaustive classical experiment search |
| `notebooks/experiments/Sweet_Spot_Threshold.ipynb` | Cleaning-threshold analysis |

## Repository Structure

```text
.
├── README.md
├── features/
├── modules/
│   ├── MODULES.md
│   ├── artifacts.py
│   ├── backbones.py
│   ├── classical_models.py
│   ├── cleaning.py
│   ├── config_types.py
│   ├── config_utils.py
│   ├── data_utils.py
│   ├── datasets.py
│   ├── deep_learning.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── grid_search.py
│   ├── image_audit.py
│   ├── threshold_experiments.py
│   ├── transforms.py
│   └── visualization.py
├── notebooks/
│   ├── 1_Traditional_Machine_Learning_Pipeline.ipynb
│   ├── 2_Deep_Learning_Pipeline.ipynb
│   └── experiments/
│       ├── Grid_Search.ipynb
│       └── Sweet_Spot_Threshold.ipynb
└── reports/
```

Generated artifacts such as cached features, figures, metrics, and trained models are produced by the notebooks during execution.

## Pipeline Overview

### 1. Hybrid Classical Pipeline

```text
Raw images
→ EDA and image audit
→ Cleaning-policy analysis
→ Stratified train/validation/test split
→ Preprocessing
→ Frozen CNN feature extraction
→ Saved feature arrays (.npy)
→ Classical classifier training and tuning
→ Validation, test evaluation, and artifact export
```

Supported classical feature backbones:

- `efficientnet_b0`
- `resnet18`
- `vgg16`

Supported classical classifiers:

- `logistic_regression`
- `svm_linear`
- `random_forest`
- `voting_soft`
- `stacking`

### 2. Deep Learning Pipeline

```text
Raw images
→ Preprocessing and augmentation
→ Transfer-learning model construction
→ Frozen-head training
→ Fine-tuning
→ Validation monitoring
→ Test evaluation and comparison
```

Supported deep learning backbones are validated in `modules/config_types.py`.

## Dataset

Recommended dataset source:

```text
tongpython/cat-and-dog
```

The notebooks are designed to obtain the dataset inside the runtime environment and should not depend on personal cloud storage paths.

## Configuration

Configuration is centralized in the `modules/` package and validated through typed config objects.

```python
from modules.config_utils import get_default_config, validate_config

config = get_default_config()
validate_config(config)
```

Main config sections:

- `dataset`
- `split`
- `cleaning`
- `preprocessing`
- `feature_extraction`
- `classifier`
- `tune-hyperparameter`
- `deep_learning`

The exact supported values and defaults are defined in `modules/config_types.py`.

## Outputs

Typical generated outputs include:

```text
features/                  cached feature arrays and manifests
reports/figures/           EDA and evaluation figures
reports/results/           metrics, summaries, and CSV outputs
models/                    trained model artifacts when saved
```

Examples of saved artifacts:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- experiment summaries in JSON/CSV format
- confusion matrices and wrong-prediction galleries
- trained classical or deep learning model files

## Evaluation

The project reports standard classification metrics:

- accuracy
- precision
- recall
- macro F1-score
- confusion matrix
- wrong-prediction analysis
- ROC-AUC when probability outputs are available

## Reported Results

The latest report compares:

- **Hybrid classical pipeline:** EfficientNet-B0 frozen features + Logistic Regression
- **Deep learning pipeline:** fine-tuned EfficientNet-B0

See `reports/` for the full report and generated figures.

## Assignment Alignment

| Requirement | Implementation |
|---|---|
| Image EDA | Class balance, image-size analysis, RGB analysis, sample visualization |
| Preprocessing | Resize, normalization, augmentation modes |
| Feature extraction | Pretrained CNN backbones with saved `.npy` features |
| Traditional ML | Logistic Regression, Linear SVM, Random Forest, Soft Voting, Stacking |
| Evaluation | Accuracy, precision, recall, macro F1, confusion matrix, wrong predictions |
| Experiments | Cleaning-threshold analysis and classical grid search |
| Extension | End-to-end deep learning transfer-learning pipeline |

## Reproducibility Notes

- Use the notebooks in the documented order.
- Keep config changes inside the config system instead of hardcoding values across cells.
- Use the same random seed when reproducing reported results.
- Generated folders may be absent before execution; the notebooks recreate them when needed.

## Module Reference

See [`modules/MODULES.md`](modules/MODULES.md) for a detailed module overview and API notes.