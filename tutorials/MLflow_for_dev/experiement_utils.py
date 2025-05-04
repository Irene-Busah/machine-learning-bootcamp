import mlflow
import mlflow.entities

# creating a function that creates an experiment
def create_experiment(name, artifact_location, tags):
    """Creates an ML experiment"""
    try:
        experiment_id = mlflow.create_experiment(
            name=name,
            artifact_location=artifact_location,
            tags=tags
        )
    
    except:
        print(f"Experiment {name} already exist")
        experiment_id = mlflow.get_experiment_by_name(name).experiment_id
    
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
