"""
This module is responsible for retrieving and splitting the dataset into training and testing sets.

Functions:
    - retrieve_train_test_data: Uses the train_test_split method to divide the dataset into training and testing subsets.

Usage:
    This module is typically used in the context of machine learning workflows where data needs to be split into 
    training and testing sets for model evaluation and validation.

Dependencies:
    - sklearn.model_selection.train_test_split: Utility function for splitting arrays or matrices into random train and test subsets.
"""


import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


def retrieve_train_test_set(data: pd.DataFrame) -> Tuple:
    """_summary_

    Args:
        data (pd.DataFrame): _description_

    Returns:
        Tuple: _description_
    """

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns="target", axis=1), data['target'], test_size=0.2, random_state=42)

    X_test, x_score, y_test, y_score = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    return X_train, X_test, x_score, y_train, y_test, y_score

