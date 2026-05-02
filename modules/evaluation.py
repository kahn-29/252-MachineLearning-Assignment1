from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _base_classifiers(seed: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=seed,
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
                        C=1.0,
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def get_classifier(name: str, seed: int = 42, **kwargs):
    """Return one configured sklearn classifier by name."""

    name = name.lower().strip()
    base = _base_classifiers(seed=seed)

    if name in base:
        model = base[name]

        if name == "logistic_regression":
            clf = model.named_steps["clf"]
            if "C" in kwargs:
                clf.C = kwargs["C"]
            if "max_iter" in kwargs:
                clf.max_iter = kwargs["max_iter"]

        elif name == "svm_linear":
            clf = model.named_steps["clf"]
            if "C" in kwargs:
                clf.C = kwargs["C"]

        elif name == "random_forest":
            if "n_estimators" in kwargs:
                model.n_estimators = kwargs["n_estimators"]
            if "max_depth" in kwargs:
                model.max_depth = kwargs["max_depth"]

        return model

    if name == "voting_soft":
        estimators = [
            ("lr", base["logistic_regression"]),
            ("svm", base["svm_linear"]),
            ("rf", base["random_forest"]),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    if name == "stacking":
        estimators = [
            ("lr", base["logistic_regression"]),
            ("svm", base["svm_linear"]),
            ("rf", base["random_forest"]),
        ]
        final_estimator = LogisticRegression(max_iter=1000, random_state=seed)
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1,
        )

    raise ValueError(
        "Unsupported classifier. Use one of: logistic_regression, svm_linear, random_forest, voting_soft, stacking."
    )


def train_classifier(X_train, y_train, classifier_name: str, seed: int = 42, **kwargs):
    """Instantiate one classifier and fit it on training data."""

    model = get_classifier(classifier_name, seed=seed, **kwargs)
    model.fit(X_train, y_train)
    return model


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Compute numeric classification metrics from true and predicted labels."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    wrong = int((y_true != y_pred).sum())
    total = int(len(y_true))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "wrong_predictions": wrong,
        "total_samples": total,
        "error_rate": float(wrong / total) if total > 0 else np.nan,
    }


def classification_report_df(y_true, y_pred, target_names=None) -> pd.DataFrame:
    """Return sklearn classification report as a dataframe."""

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_df(y_true, y_pred, labels=None, label_names=None) -> pd.DataFrame:
    """Return confusion matrix as a dataframe table."""

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if label_names is None:
        if labels is None:
            unique = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
            row_names = [f"Actual {x}" for x in unique]
            col_names = [f"Predicted {x}" for x in unique]
        else:
            row_names = [f"Actual {x}" for x in labels]
            col_names = [f"Predicted {x}" for x in labels]
    else:
        row_names = [f"Actual {name}" for name in label_names]
        col_names = [f"Predicted {name}" for name in label_names]

    return pd.DataFrame(cm, index=row_names, columns=col_names)


def evaluate_model(model, X, y, model_name: str = "model") -> dict:
    """Predict on X and return model name with computed metrics."""

    y_pred = model.predict(X)
    metrics = compute_classification_metrics(y, y_pred)
    return {"model": model_name, **metrics}


def find_wrong_predictions(
    df: pd.DataFrame,
    y_true,
    y_pred,
    path_col: str = "path",
    label_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Return misclassified rows with true/pred labels and optional names."""

    if label_map is None:
        label_map = {0: "Cat", 1: "Dog"}

    output = df.copy().reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(output) != len(y_true):
        raise ValueError(f"Length mismatch: dataframe has {len(output)} rows but y_true has {len(y_true)}.")

    output["true_label"] = y_true
    output["pred_label"] = y_pred
    output["true_label_name"] = output["true_label"].map(label_map)
    output["pred_label_name"] = output["pred_label"].map(label_map)
    output["is_correct"] = output["true_label"] == output["pred_label"]

    keep_cols = [
        col
        for col in [
            path_col,
            "true_label",
            "pred_label",
            "true_label_name",
            "pred_label_name",
            "is_correct",
        ]
        if col in output.columns
    ]

    return output.loc[~output["is_correct"], keep_cols].reset_index(drop=True)
