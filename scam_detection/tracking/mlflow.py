from pathlib import Path

import mlflow


def setup_mlflow_tracking(mlflow_tracking_uri: str = "http://localhost:5000"):
    import requests
    from requests.exceptions import RequestException

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if mlflow_tracking_uri.startswith("http://") or mlflow_tracking_uri.startswith(
        "https://"
    ):
        try:
            print(
                f"Attempting to connect to MLflow server at "
                f"{mlflow_tracking_uri}..."
            )
            response = requests.get(f"{mlflow_tracking_uri}/health", timeout=5)
            if response.status_code == 200:
                print(
                    f"✓ Successfully connected to MLflow server at "
                    f"{mlflow_tracking_uri}"
                )
            else:
                print(
                    f"⚠ Warning: MLflow server at {mlflow_tracking_uri} "
                    f"returned status {response.status_code}"
                )
                print("   Falling back to local filesystem tracking (./mlruns)")
        except RequestException as e:
            print(
                f"✗ ERROR: Cannot connect to MLflow server at " f"{mlflow_tracking_uri}"
            )
            print(f"   Error: {e}")
            print("   Please ensure MLflow server is running:")
            port = mlflow_tracking_uri.split(":")[-1]
            print(f"   mlflow server --host 127.0.0.1 --port {port}")
            print("   Falling back to local filesystem tracking (./mlruns)")
            mlflow.set_tracking_uri("./mlruns")


def ensure_experiment_exists(experiment_name: str):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(
                f"Created MLflow experiment '{experiment_name}' "
                f"with ID: {experiment_id}"
            )
        else:
            print(
                f"Using existing MLflow experiment '{experiment_name}' "
                f"with ID: {experiment.experiment_id}"
            )
    except Exception as e:
        print(f"Warning: Could not ensure experiment exists: {e}")
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(
                f"Created MLflow experiment '{experiment_name}' "
                f"with ID: {experiment_id} after initial error"
            )
        except Exception as e2:
            print(f"Failed to create experiment: {e2}")


def log_git_commit():
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            mlflow.log_param("git_commit", commit_hash)
        else:
            mlflow.log_param("git_commit", "unknown")
    except Exception:
        mlflow.log_param("git_commit", "unknown")
