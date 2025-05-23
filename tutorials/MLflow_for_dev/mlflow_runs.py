import mlflow


if __name__ == "__main__":
    with mlflow.start_run(run_name="mlflow_run") as run:
        mlflow.log_param("learning_rate", 0.01)

        print(f"Run ID: {run.info.run_id}\n")
        print(run.info)
