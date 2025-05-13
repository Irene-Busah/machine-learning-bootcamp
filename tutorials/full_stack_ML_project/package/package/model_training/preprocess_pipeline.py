"""_summary_
This module contains functions to create and manage a machine learning preprocessing pipeline.

The preprocessing pipeline is responsible for preparing raw data for machine learning models.
It may include steps such as handling missing values, encoding categorical variables, scaling
numerical features, and other transformations required for model training.

Functions in this module are designed to streamline the preprocessing workflow and ensure
consistency across different datasets and experiments.
"""

# importing necessary libraries
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def get_pipeline(numerical_features: List[str], categorical_features: List[str]):
    
    """
        Creates a preprocessing pipeline for numerical and categorical features.
            numerical_features (List[str]): A list of column names representing numerical features.
            categorical_features (List[str]): A list of column names representing categorical features.
        Returns:
            sklearn.pipeline.Pipeline: A pipeline object that preprocesses the input features.
    """

    transformer = ColumnTransformer(
        [
            ("numerical_imputer", SimpleImputer(strategy="median"), numerical_features),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    pipeline = Pipeline([("transfomer", transformer), ("Classifier", RandomForestClassifier())])

    return pipeline
