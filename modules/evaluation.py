# modules/evaluation.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def get_base_classifiers(seed: int = 42) -> dict[str, Any]:
    """
    Return base classifiers used in the hybrid pipeline.

    Returns
    -------
    dict
        Dictionary of classifier name -> sklearn model.
    """

    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        max_iter=1000,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        C=0.1,
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def get_classifier(
    name: str,
    seed: int = 42,
    **kwargs,
):
    """
    Return a sklearn classifier or ensemble.

    Supported classifiers:
    - logistic_regression
    - svm_linear
    - random_forest
    - voting_soft
    - stacking

    Parameters
    ----------
    name : str
        Classifier name.
    seed : int
        Random seed.
    **kwargs
        Optional hyperparameter overrides.

    Returns
    -------
    sklearn estimator
        Classifier object.
    """

    name = name.lower().strip()
    base = get_base_classifiers(seed=seed)

    if name in base:
        model = base[name]

        if name == "logistic_regression":
            if "C" in kwargs:
                model.named_steps["clf"].C = kwargs["C"]
            if "max_iter" in kwargs:
                model.named_steps["clf"].max_iter = kwargs["max_iter"]

        elif name == "svm_linear":
            if "C" in kwargs:
                model.named_steps["clf"].C = kwargs["C"]

        elif name == "random_forest":
            if "n_estimators" in kwargs:
                model.n_estimators = kwargs["n_estimators"]

        return model

    if name == "voting_soft":
        estimators = [
            ("lr", base["logistic_regression"]),
            ("svm", base["svm_linear"]),
            ("rf", base["random_forest"]),
        ]

        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        )

    if name == "stacking":
        estimators = [
            ("lr", base["logistic_regression"]),
            ("svm", base["svm_linear"]),
            ("rf", base["random_forest"]),
        ]

        final_estimator = LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=seed,
        )

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported classifier: {name}. "
        "Supported classifiers: logistic_regression, svm_linear, "
        "random_forest, voting_soft, stacking."
    )


def train_classifier(
    X_train,
    y_train,
    classifier_name: str,
    seed: int = 42,
    **kwargs,
):
    """
    Train one selected classifier.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    classifier_name : str
        Classifier name.
    seed : int
        Random seed.
    **kwargs
        Optional classifier parameters.

    Returns
    -------
    sklearn estimator
        Trained classifier.
    """

    model = get_classifier(
        name=classifier_name,
        seed=seed,
        **kwargs,
    )

    model.fit(X_train, y_train)

    return model


def train_classifier_candidates(
    X_train,
    y_train,
    classifier_names: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train multiple classifiers.

    This function is useful when ensemble or comparison mode is enabled.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    classifier_names : list[str]
        List of classifier names.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary of classifier name -> trained model.
    """

    models = {}

    for name in classifier_names:
        model = train_classifier(
            X_train=X_train,
            y_train=y_train,
            classifier_name=name,
            seed=seed,
        )

        models[name] = model

    return models


def predict_model(model, X):
    """
    Predict class labels using a trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """

    return model.predict(X)


def predict_proba_safe(model, X):
    """
    Predict class probabilities if supported.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray | None
        Predicted probabilities, or None if unavailable.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    return None


def compute_classification_metrics(
    y_true,
    y_pred,
) -> dict[str, Any]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Classification metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    wrong = int((y_true != y_pred).sum())

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "wrong_predictions": wrong,
        "total_samples": int(len(y_true)),
        "error_rate": float(wrong / len(y_true)) if len(y_true) > 0 else np.nan,
    }

    return metrics


def classification_report_df(
    y_true,
    y_pred,
    target_names=("Cat", "Dog"),
) -> pd.DataFrame:
    """
    Return sklearn classification report as dataframe.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    target_names : tuple
        Class names.

    Returns
    -------
    pd.DataFrame
        Classification report dataframe.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(target_names),
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).T


def confusion_matrix_df(
    y_true,
    y_pred,
    labels=(0, 1),
    label_names=("Cat", "Dog"),
) -> pd.DataFrame:
    """
    Return confusion matrix as dataframe.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : tuple
        Numeric labels.
    label_names : tuple
        Label names.

    Returns
    -------
    pd.DataFrame
        Confusion matrix dataframe.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(labels),
    )

    return pd.DataFrame(
        cm,
        index=[f"Actual {name}" for name in label_names],
        columns=[f"Predicted {name}" for name in label_names],
    )


def evaluate_model(
    model,
    X,
    y,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Evaluate one trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X : array-like
        Feature matrix.
    y : array-like
        Ground-truth labels.
    model_name : str
        Name used in output dictionary.

    Returns
    -------
    dict
        Evaluation results.
    """

    y_pred = predict_model(model, X)
    metrics = compute_classification_metrics(y, y_pred)

    result = {
        "model": model_name,
        **metrics,
    }

    return result


def evaluate_models(
    models: dict[str, Any],
    X_val,
    y_val,
) -> pd.DataFrame:
    """
    Evaluate multiple trained models.

    Parameters
    ----------
    models : dict
        Dictionary of model name -> trained model.
    X_val : array-like
        Validation features.
    y_val : array-like
        Validation labels.

    Returns
    -------
    pd.DataFrame
        Model comparison table.
    """

    records = []

    for name, model in models.items():
        result = evaluate_model(
            model=model,
            X=X_val,
            y=y_val,
            model_name=name,
        )
        records.append(result)

    return pd.DataFrame(records).sort_values(
        by="f1_macro",
        ascending=False,
    ).reset_index(drop=True)


def select_best_model(
    validation_df: pd.DataFrame,
    metric: str = "f1_macro",
) -> str:
    """
    Select the best model based on validation results.

    Parameters
    ----------
    validation_df : pd.DataFrame
        Validation result dataframe.
    metric : str
        Metric used for model selection.

    Returns
    -------
    str
        Best model name.
    """

    if metric not in validation_df.columns:
        raise KeyError(f"Metric not found in validation_df: {metric}")

    if "model" not in validation_df.columns:
        raise KeyError("validation_df must contain a 'model' column.")

    best_row = validation_df.sort_values(
        by=metric,
        ascending=False,
    ).iloc[0]

    return str(best_row["model"])


def find_wrong_predictions(
    df: pd.DataFrame,
    y_true,
    y_pred,
    path_col: str = "path",
    label_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Return dataframe of misclassified samples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe corresponding to predictions.
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    path_col : str
        Image path column.
    label_names : dict[int, str] | None
        Mapping from numeric label to class name.

    Returns
    -------
    pd.DataFrame
        Wrong prediction dataframe.
    """

    if label_names is None:
        label_names = {
            0: "Cat",
            1: "Dog",
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    output = df.copy().reset_index(drop=True)

    if len(output) != len(y_true):
        raise ValueError(
            f"Length mismatch: df has {len(output)} rows, "
            f"but y_true has {len(y_true)} elements."
        )

    output["true_label"] = y_true
    output["pred_label"] = y_pred
    output["true_label_name"] = output["true_label"].map(label_names)
    output["pred_label_name"] = output["pred_label"].map(label_names)
    output["is_correct"] = output["true_label"] == output["pred_label"]

    columns = [
        c for c in [
            path_col,
            "true_label",
            "pred_label",
            "true_label_name",
            "pred_label_name",
            "is_correct",
        ]
        if c in output.columns
    ]

    wrong_df = output[~output["is_correct"]].copy()

    return wrong_df[columns].reset_index(drop=True)


def compare_result_tables(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Combine multiple result dictionaries into one comparison dataframe.

    Parameters
    ----------
    results : list[dict]
        List of result dictionaries.

    Returns
    -------
    pd.DataFrame
        Comparison dataframe.
    """

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    preferred_cols = [
        "model",
        "accuracy",
        "f1_macro",
        "wrong_predictions",
        "total_samples",
        "training_strategy",
        "training_time_seconds",
    ]

    ordered_cols = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols + remaining_cols]


def save_metrics_json(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save metrics dictionary to JSON.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    output_path : str | Path
        Output path.

    Returns
    -------
    Path
        Saved path.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}

    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            serializable[key] = int(value)
        elif isinstance(value, (np.floating,)):
            serializable[key] = float(value)
        elif isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)

    return output_path


def load_metrics_json(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load metrics dictionary from JSON.

    Parameters
    ----------
    path : str | Path
        JSON path.

    Returns
    -------
    dict
        Loaded metrics.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latex_ready_metrics(
    metrics: dict[str, Any],
    digits: int = 4,
) -> str:
    """
    Create a LaTeX/report-ready metrics summary.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    digits : int
        Number of decimal digits.

    Returns
    -------
    str
        Formatted metrics text.
    """

    lines = []

    if "accuracy" in metrics:
        lines.append(f"Accuracy: {metrics['accuracy']:.{digits}f}")

    if "precision_macro" in metrics:
        lines.append(f"Macro precision: {metrics['precision_macro']:.{digits}f}")

    if "recall_macro" in metrics:
        lines.append(f"Macro recall: {metrics['recall_macro']:.{digits}f}")

    if "f1_macro" in metrics:
        lines.append(f"Macro F1-score: {metrics['f1_macro']:.{digits}f}")

    if "wrong_predictions" in metrics and "total_samples" in metrics:
        lines.append(
            f"Incorrect predictions: "
            f"{metrics['wrong_predictions']} / {metrics['total_samples']}"
        )

    if "training_time_seconds" in metrics:
        lines.append(f"Training time: {metrics['training_time_seconds']:.2f} seconds")

    return "\n".join(lines)


def confusion_matrix_latex_values(
    y_true,
    y_pred,
    labels=(0, 1),
) -> dict[str, int]:
    """
    Return confusion matrix values for LaTeX tables.

    For binary Cat-Dog setting:
    - cat_cat
    - cat_dog
    - dog_cat
    - dog_dog

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : tuple
        Label order.

    Returns
    -------
    dict
        Confusion matrix values.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(labels),
    )

    if cm.shape != (2, 2):
        raise ValueError("confusion_matrix_latex_values expects binary classification.")

    return {
        "cat_cat": int(cm[0, 0]),
        "cat_dog": int(cm[0, 1]),
        "dog_cat": int(cm[1, 0]),
        "dog_dog": int(cm[1, 1]),
    }


def print_latex_ready_output(
    metrics: dict[str, Any],
    cm_values: dict[str, int] | None = None,
) -> None:
    """
    Print LaTeX-ready values for report writing.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    cm_values : dict | None
        Confusion matrix value dictionary.
    """

    print("LaTeX-ready values")
    print("------------------")
    print(latex_ready_metrics(metrics))

    if cm_values is not None:
        print()
        print("Confusion matrix values:")
        print(
            f"Actual Cat -> Pred Cat: {cm_values['cat_cat']}, "
            f"Pred Dog: {cm_values['cat_dog']}"
        )
        print(
            f"Actual Dog -> Pred Cat: {cm_values['dog_cat']}, "
            f"Pred Dog: {cm_values['dog_dog']}"
        )