# importing the necessary functions
import json
import requests
import mlflow.sklearn
from package.package.feature.data_processing import get_features_dataframe
from package.package.model_training.train_test_data_retrieval import retrieve_train_test_set

from sklearn.metrics import classification_report
import pandas as pd
import mlflow


if __name__ == "__main__":
    data = get_features_dataframe()

    x_train, x_test, x_score, y_train, y_test, y_score = retrieve_train_test_set(data)

    features = [feature for feature in x_train.columns if feature not in ["id", "target", "MedHouseVal"]]

    print(features)

    feature_values = json.loads(x_score[features].iloc[1:2].to_json(orient="split"))

    payload = {"dataframe_split": feature_values}

    print(payload)

    model_uri = "models:/registered_model/latest"
    
    BASE_URI = "http://localhost:5000/"

    headers = {"Content-Type": "application/json"}
    endpoint = BASE_URI + "invocations"

    request = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    print(f"STATUS CODE: {request.status_code}")
    print(f"PREDICTIONS: {request.text}")
    print(f"TARGET: {y_score.iloc[1:2]}")


