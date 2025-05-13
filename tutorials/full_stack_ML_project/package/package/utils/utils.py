from typing import Dict
import mlflow
import mlflow.entities
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# creating a function that creates an experiment
def set_or_create_experiment(name):
    """Creates an ML experiment"""
    try:
        experiment_id = mlflow.get_experiment_by_name(name=name).experiment_id
    
    except:
        experiment_id = mlflow.create_experiment(name=name)
    finally:
        mlflow.set_experiment(name)
    
    return experiment_id

# function to retrieve experiment
def retrieve_experiment(experiment_id=None, experiment_name=None) -> mlflow.entities.Experiment:
    """Retrieves an experiment using ID or name"""

    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment id or experiment name must be provided.")
    
    return experiment

# function to get the performance metrics
def get_performance_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str):
    """
    Generates performance metric visualizations for a classification model and returns them as a dictionary.
    y_true (pd.DataFrame): Ground truth (actual) target values.
    y_pred (pd.DataFrame): Predicted target values from the model.
    prefix (str): A prefix string to be used as a key identifier for the returned figures.
    Returns:
        dict: A dictionary containing the following matplotlib figures:
            - "<prefix>_roc_curve": ROC curve figure.
            - "<prefix>_confusion_matrix": Confusion matrix figure.
            - "<prefix>_precision_recall_curve": Precision-recall curve figure.
    """

    roc_figure = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    confusion_matrix_figure = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    precision_recall_figure = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    return {
        f"{prefix}_roc_curve": roc_figure,
        f"{prefix}_confusion_matrix": confusion_matrix_figure,
        f"{prefix}_precision_recall_curve": precision_recall_figure
    }


# function to compute the classification scores of the model
def classification_scores(y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """
    Computes and logs classification evaluation metrics for a given set of true 
    and predicted labels, with an optional prefix for metric names.
        y_true (pd.DataFrame): A DataFrame containing the true labels.
        y_pred (pd.DataFrame): A DataFrame containing the predicted labels.
        prefix (str): A string prefix to prepend to the metric names for logging.
    Returns:
        Dict[str, float]: A dictionary containing the computed metrics.
    """
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1_score": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred)
    }
    return metrics


