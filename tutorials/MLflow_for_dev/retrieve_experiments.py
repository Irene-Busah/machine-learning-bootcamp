from experiement_utils import retrieve_experiment


if __name__ == "__main__":
    experiment = retrieve_experiment(experiment_name="testing_mlflow")

    print(f"Name: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    print(f"Creation Timestamp: {experiment.creation_time}")
