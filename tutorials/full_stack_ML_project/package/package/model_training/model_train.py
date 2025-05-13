"""
    This module provides functionality for training machine learning models as part of a 
    full-stack machine learning project. It includes functions and classes that facilitate 
    the training process, such as data preprocessing, model initialization, training, 
    evaluation, and saving the trained models.
    The module is designed to be modular and reusable, allowing for easy integration 
    into larger machine learning pipelines. It supports various machine learning 
    frameworks and can be extended to accommodate custom training workflows.

    Author:
        Irene Busah
"""


# import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
import mlflow

from typing import Tuple


def train_model(pipeline: Pipeline, run_name: str, x:pd.DataFrame, y:pd.DataFrame) -> Tuple[str, Pipeline]:
    """
        Trains a machine learning model using the provided pipeline and training data.
        pipeline (Pipeline): A scikit-learn pipeline object that defines the sequence of 
        transformations and the model to be trained.
        run_name (str): A unique name for the training run, used for logging or tracking purposes.
        x: The input features for training the model. This should be a pandas 
            DataFrame where each row represents a training example and each column represents a feature.
        y: The target labels corresponding to the input features. This should be a 
            pandas DataFrame or Series where each row corresponds to the target value for the 
            respective training example in `x`.
        Tuple[str, Pipeline]: A tuple containing:
            - str: The name of the training run, which can be used for tracking or logging.
            - Pipeline: The trained scikit-learn pipeline object, which includes the fitted 
              transformations and the trained model.
    """

    signature = infer_signature(x, y)
    with mlflow.start_run(run_name=run_name) as run:
        pipeline = pipeline.fit(x, y)
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", signature=signature)
    
    return run.info.run_id, pipeline

