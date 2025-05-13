# importing the necessary functions
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

    model_uri = "models:/registered_model/latest"
    model = mlflow.sklearn.load_model(model_uri=model_uri)

    predictions = model.predict(x_score[features])
    scored_data = pd.DataFrame({"prediction": predictions, "target": y_score})

    report = classification_report(y_score, predictions)
    print(report)
    print(scored_data.head(10))

