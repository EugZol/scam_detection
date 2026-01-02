"""
MLflow utilities for experiment tracking.
"""

from pathlib import Path

import mlflow


def setup_mlflow_tracking(mlflow_tracking_uri: str = "http://localhost:5000"):
    """Set up MLflow tracking URI."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)


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
