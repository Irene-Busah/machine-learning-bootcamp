"""
    Description:
        This module provides functionality for loading and preprocessing the California housing dataset.
        It includes methods to retrieve the dataset as a pandas DataFrame and to generate a features
        DataFrame with additional columns for unique identifiers and binary target labels.
    Functions:
        - load_dataset() -> pd.DataFrame:
            Downloads the California housing dataset from sklearn's dataset repository and returns it
            as a pandas DataFrame. The dataset is cached locally in a 'data' directory within the
            current file's directory.
        - get_features_dataframe() -> pd.DataFrame:
            Processes the California housing dataset to add an 'id' column (based on the index) and a
            binary 'target' column. The 'target' column is derived by comparing the 'MedHouseVal' column
            to its median value, where values greater than or equal to the median are labeled as 1, and
            others as 0.
    Usage:
        This module is intended to be used as part of a machine learning pipeline for data retrieval
        and preprocessing. The `load_dataset` function provides the raw dataset, while the
        `get_features_dataframe` function prepares the data for further analysis or modeling.
    Dependencies:
        - pandas (pd): Used for handling tabular data.
        - sklearn.datasets.fetch_california_housing: Used to fetch the California housing dataset.
        - os: Used for handling file paths and directory operations.
    Notes:
        - The dataset is downloaded and cached locally to avoid repeated downloads.
        - The binary 'target' column is created for classification tasks, where the target is based on
          whether the median house value is above or below the median value of the dataset.
    Author:
        Irene Busah
    Created On:
        11th May 2025
"""

from sklearn.datasets import fetch_california_housing
import pandas as pd
import os


def load_dataset() -> pd.DataFrame:
    """Downloads the California housing dataset and returns it as a dataframe

    Returns:
        pd.DataFrame: California Housing dataset as a pandas dataframe
    """

    dir_path = os.path.abspath(os.path.dirname(__file__))
    data = fetch_california_housing(data_home=f"{dir_path}/data/", as_frame=True, download_if_missing=True)

    return data.frame


def get_features_dataframe() -> pd.DataFrame:
    """Retrieves the features dataframe

    Returns:
        pd.DataFrame: A Features dataframe
    """

    data = load_dataset()
    data['id'] = data.index
    data['target'] = data['MedHouseVal'] >= data['MedHouseVal'].median()
    data['target'] = data['target'].astype(int)

    return data

