"""
MLflow utilities for experiment tracking.
"""

from pathlib import Path

import mlflow


def setup_mlflow_tracking(mlflow_tracking_uri: str = "http://localhost:5000"):
    """Set up MLflow tracking URI with connection verification."""
    import requests
    from requests.exceptions import RequestException

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Verify connection if using remote tracking server
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
            # Fall back to local tracking
            mlflow.set_tracking_uri("./mlruns")


def ensure_experiment_exists(experiment_name: str):
    """Ensure MLflow experiment exists, create if it doesn't."""
    try:
        # Try to get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create the experiment if it doesn't exist
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
        # Try to create the experiment anyway
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(
                f"Created MLflow experiment '{experiment_name}' "
                f"with ID: {experiment_id} after initial error"
            )
        except Exception as e2:
            print(f"Failed to create experiment: {e2}")


def cleanup_malformed_experiments():
    """Clean up malformed experiments in MLflow."""
    try:
        import shutil
        from pathlib import Path

        # Get MLflow tracking URI
        tracking_uri = mlflow.get_tracking_uri()

        # Check both local mlruns and the tracking URI location
        mlruns_paths = []
        if "file:" in tracking_uri or not tracking_uri.startswith("http"):
            # Local filesystem tracking
            mlruns_path = Path(tracking_uri.replace("file:", "").replace("file://", ""))
            mlruns_paths.append(mlruns_path)

        # Always check local mlruns directory too
        local_mlruns = Path("./mlruns")
        if local_mlruns not in mlruns_paths:
            mlruns_paths.append(local_mlruns)

        for mlruns_path in mlruns_paths:
            if mlruns_path.exists():
                print(f"Checking for malformed experiments in {mlruns_path}")
                for exp_dir in mlruns_path.iterdir():
                    if exp_dir.is_dir() and exp_dir.name.isdigit():
                        meta_file = exp_dir / "meta.yaml"
                        if not meta_file.exists():
                            print(f"Removing malformed experiment directory: {exp_dir}")
                            shutil.rmtree(exp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not cleanup malformed experiments: {e}")


def log_git_commit():
    """Log the current git commit hash to MLflow."""
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
