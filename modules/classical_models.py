from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


_SUPPORTED_CLASSIFIERS = (
    "logistic_regression",
    "svm_linear",
    "random_forest",
    "voting_soft",
    "stacking",
)


def list_supported_classifiers() -> list[str]:
    """Return supported classical classifier names."""
    return list(_SUPPORTED_CLASSIFIERS)


def _normalize_classifier_name(name: str) -> str:
    """Normalize and validate classifier name."""
    normalized = name.lower().strip()
    if normalized not in _SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unsupported classifier: {name}. "
            f"Supported classifiers: {list_supported_classifiers()}"
        )
    return normalized

def _build_logistic_regression(seed: int, **params: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            class_weight=params.get("class_weight"),
            solver=params.get("solver", "lbfgs"),
            random_state=seed,
        )),
    ])


def _build_svm_linear(seed: int, **params: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="linear",
            C=params.get("C", 1.0),
            probability=params.get("probability", True),
            class_weight=params.get("class_weight"),
            random_state=seed,
        )),
    ])


def _build_random_forest(seed: int, **params: Any) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        class_weight=params.get("class_weight"),
        random_state=seed,
        n_jobs=params.get("n_jobs", -1),
    )


def get_classifier(name: str, seed: int = 42, **params: Any) -> BaseEstimator:
    """Build a classical classifier from a name and parameter dictionary."""
    name = _normalize_classifier_name(name)

    if name == "logistic_regression":
        return _build_logistic_regression(seed=seed, **params)

    if name == "svm_linear":
        return _build_svm_linear(seed=seed, **params)

    if name == "random_forest":
        return _build_random_forest(seed=seed, **params)

    if name == "voting_soft":
        return VotingClassifier(
            estimators=[
                ("lr", _build_logistic_regression(seed=seed, **params.get("lr", {}))),
                ("svm", _build_svm_linear(seed=seed, **params.get("svm", {}))),
                ("rf", _build_random_forest(seed=seed, **params.get("rf", {}))),
            ],
            voting=params.get("voting", "soft"),
            n_jobs=params.get("n_jobs", -1),
        )

    if name == "stacking":
        return StackingClassifier(
            estimators=[
                ("lr", _build_logistic_regression(seed=seed, **params.get("lr", {}))),
                ("svm", _build_svm_linear(seed=seed, **params.get("svm", {}))),
                ("rf", _build_random_forest(seed=seed, **params.get("rf", {}))),
            ],
            final_estimator=LogisticRegression(
                max_iter=params.get("final_max_iter", 1000),
                random_state=seed,
            ),
            n_jobs=params.get("n_jobs", -1),
        )

    raise ValueError(f"Unsupported classifier: {name}")


def train_classifier(
    X_train,
    y_train,
    classifier_name: str,
    seed: int = 42,
    **params: Any,
) -> BaseEstimator:
    """Train a classical classifier on training data."""
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must not be None.")

    model = get_classifier(classifier_name, seed=seed, **params)
    model.fit(X_train, y_train)
    return model

def _classifier_run_name(cfg: dict, index: int) -> str:
    """Build a stable display/run name for a classifier config."""
    if "run_name" in cfg:
        return str(cfg["run_name"])

    name = str(cfg["name"])
    params = cfg.get("params", {})

    if not params:
        return f"{index:02d}_{name}"

    short_params = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
    return f"{index:02d}_{name}_{short_params}"


def benchmark_classifiers(
    X_train,
    y_train,
    X_val,
    y_val,
    classifier_configs: list[dict],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Train multiple classifiers and compute validation metrics."""
    if not classifier_configs:
        raise ValueError("classifier_configs must contain at least one classifier config.")

    rows: list[dict[str, Any]] = []
    trained: dict[str, BaseEstimator] = {}

    for index, cfg in enumerate(classifier_configs):
        if "name" not in cfg:
            raise KeyError(f"Missing 'name' in classifier config at index {index}: {cfg}")

        name = str(cfg["name"])
        params = dict(cfg.get("params", {}))
        run_name = _classifier_run_name(cfg, index)

        model = train_classifier(X_train, y_train, name, seed=seed, **params)
        preds = model.predict(X_val)

        trained[run_name] = model
        rows.append({
            "run_name": run_name,
            "model": name,
            "params": params,
            "accuracy": accuracy_score(y_val, preds),
            "precision_macro": precision_score(y_val, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, preds, average="macro", zero_division=0),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True),
        trained,
    )


def select_best_model(
    results_df: pd.DataFrame,
    trained_models: dict[str, BaseEstimator],
    metric: str = "f1_macro",
    model_key_col: str = "run_name",
) -> tuple[str, BaseEstimator, pd.Series]:
    """Select the best trained model from benchmark results."""
    if results_df.empty:
        raise ValueError("results_df is empty; cannot select best model.")

    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results_df columns: {list(results_df.columns)}")

    if model_key_col not in results_df.columns:
        raise ValueError(f"Model key column '{model_key_col}' not found in results_df.")

    best_row = results_df.sort_values(metric, ascending=False).iloc[0]
    model_key = str(best_row[model_key_col])

    if model_key not in trained_models:
        raise KeyError(f"Model '{model_key}' not found in trained_models.")

    return model_key, trained_models[model_key], best_row


def _normalize_param_grid_for_estimator(
    classifier_name: str,
    param_grid: dict[str, Any],
) -> dict[str, Any]:
    """Map simple classifier params to sklearn Pipeline params when needed."""
    name = _normalize_classifier_name(classifier_name)

    if name in {"logistic_regression", "svm_linear"}:
        return {
            key if "__" in key else f"clf__{key}": value
            for key, value in param_grid.items()
        }

    return param_grid


def tune_classifier_grid(
    X_train,
    y_train,
    classifier_name: str,
    param_grid: dict[str, Any],
    cv: int = 3,
    seed: int = 42,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
) -> GridSearchCV:
    """Perform grid search hyperparameter tuning on a classifier."""
    if not param_grid:
        raise ValueError("param_grid must not be empty.")

    estimator = get_classifier(classifier_name, seed=seed)
    normalized_grid = _normalize_param_grid_for_estimator(classifier_name, param_grid)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=normalized_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    return search