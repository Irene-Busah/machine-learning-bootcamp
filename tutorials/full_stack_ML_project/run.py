"""
    run.py

    This script serves as the entry point for executing the full stack machine learning project. 
    It is responsible for orchestrating the execution of various components of the project, 
    such as data preprocessing, model training, evaluation, and deployment.

    The script is designed to provide a seamless way to run the entire machine learning pipeline 
    or specific parts of it, depending on the implementation. It acts as a central hub for 
    coordinating the workflow of the project.

    Key Responsibilities:
        - Initialize and configure the project environment.
        - Trigger data preprocessing steps, such as data cleaning and feature engineering.
        - Execute model training and hyperparameter tuning processes.
        - Evaluate the trained model on test data and generate performance metrics.
        - Optionally deploy the trained model to a production environment or save it for later use.

    Usage:
        To run the full pipeline or specific components, execute this script using Python:
            python run.py

        Depending on the implementation, additional command-line arguments or configuration 
        files may be required to specify the behavior of the script.

    Notes:
        - Ensure that all dependencies and required files are properly set up before running this script.
        - Refer to the project documentation for detailed instructions on configuring and using this script.
"""

import mlflow
from package.package.feature.data_processing import get_features_dataframe
from package.package.model_training.train_test_data_retrieval import retrieve_train_test_set
from package.package.model_training.preprocess_pipeline import get_pipeline
from package.package.utils.utils import set_or_create_experiment, get_performance_metrics, classification_scores
from package.package.model_training.model_train import train_model


if __name__ == '__main__':
    experiment_name = "house_pricing_classifier"
    run_name = "training_classifier"
    artifact_path = "model"
    model_name = "registered_model"
    data = get_features_dataframe()

    # print(data.head(5))

    # splitting the dataset 
    x_train, x_test, x_score, y_train, y_test, y_score = retrieve_train_test_set(data)

    # getting the features
    features = [feature for feature in x_train.columns if feature not in ["id", "target", "MedHouseVal"]]

    # creating the pipeline
    pipeline = get_pipeline(numerical_features=features, categorical_features=[])

    # retrieving the experiment id
    experiment_id = set_or_create_experiment(name=experiment_name)

    run_id, model = train_model(pipeline=pipeline, run_name=run_name, x=x_train, y=y_train)

    # building the ML model
    y_pred = model.predict(x_test)

    # classification metrics
    metrics = classification_scores(y_true=y_test, y_pred=y_pred, prefix="test")

    performance_plots = get_performance_metrics(y_test, y_pred, prefix="test")

    mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name)

    print("Starting the MLflow Experiment Run.....\n")

    with mlflow.start_run(run_id=run_id):
        # logging the metrics
        mlflow.log_metrics(metrics)

        # logging the parameter
        mlflow.log_params(model[-1].get_params())

        # logging the tags
        mlflow.set_tags({"model_type": "classifier", "author": "Irene Busah"})

        # logging the description
        mlflow.set_tag("mlflow.note.content", "This is a classifier for the house pricing dataset")

        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, plot_name+".png")

    print("\nSuccessfully Completed!")