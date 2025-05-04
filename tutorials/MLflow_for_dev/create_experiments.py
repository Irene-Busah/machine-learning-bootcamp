import mlflow
from experiement_utils import create_experiment


if __name__ == '__main__':
    # mlflow.create_experiment(
    #     name="testing_mlflow",
    #     artifact_location='testing_mlflow1_artifacts',
    #     tags={"env": "dev", "version": "1.0.0"}
    # )

    create_experiment("testing_mlflow1", "testing_mlflow1_artifacts", {"env": "dev", "version": "1.0.0"})

