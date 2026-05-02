import numpy as np
import pandas as pd


def get_classifier(
    name: str,
    seed: int = 42,
    **kwargs
):
    """
    Return sklearn classifier or pipeline.
    Supported:
    logistic_regression,
    svm_linear,
    random_forest,
    voting_soft,
    stacking.
    """
    ...


def get_base_classifiers(seed: int = 42) -> dict:
    """
    Return base classifiers:
    logistic_regression, svm_linear, random_forest.
    """
    ...


def train_classifier(
    X_train,
    y_train,
    classifier_name: str,
    seed: int = 42,
    **kwargs
):
    """
    Train one selected classifier.
    """
    ...


def train_classifier_candidates(
    X_train,
    y_train,
    classifier_names: list[str],
    seed: int = 42
) -> dict:
    """
    Train multiple classifiers.
    Used when ensemble/comparison mode is enabled.
    """
    ...


def predict_model(model, X):
    """
    Return predicted labels.
    """
    ...


def predict_proba_safe(model, X):
    """
    Return probabilities if supported.
    Otherwise return None.
    """
    ...


def compute_classification_metrics(
    y_true,
    y_pred
) -> dict:
    """
    Compute accuracy, macro precision, macro recall,
    macro F1, and wrong prediction count.
    """
    ...


def classification_report_df(
    y_true,
    y_pred,
    target_names=("Cat", "Dog")
) -> pd.DataFrame:
    """
    Return sklearn classification report as dataframe.
    """
    ...


def confusion_matrix_df(
    y_true,
    y_pred,
    labels=(0, 1),
    label_names=("Cat", "Dog")
) -> pd.DataFrame:
    """
    Return confusion matrix as dataframe.
    """
    ...


def evaluate_model(
    model,
    X,
    y,
    model_name: str = "model"
) -> dict:
    """
    Predict and compute metrics for one model.
    """
    ...


def evaluate_models(
    models: dict,
    X_val,
    y_val
) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison dataframe.
    """
    ...


def select_best_model(
    validation_df: pd.DataFrame,
    metric: str = "f1_macro"
) -> str:
    """
    Return model name with best validation metric.
    """
    ...


def find_wrong_predictions(
    df: pd.DataFrame,
    y_true,
    y_pred,
    path_col: str = "path"
) -> pd.DataFrame:
    """
    Return dataframe of misclassified samples with true/pred labels.
    """
    ...


def compare_result_tables(
    results: list[dict]
) -> pd.DataFrame:
    """
    Combine multiple result dictionaries into one comparison dataframe.
    Useful for hybrid vs end-to-end DL comparison.
    """
    ...


def latex_ready_metrics(metrics: dict) -> str:
    """
    Return formatted text containing LaTeX-ready values.
    """
    ...