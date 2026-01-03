from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .data.datamodule import EmailDataModule
from .training.trainer import train_tfidf_model, train_transformer_model


def train(cfg: DictConfig):
    """
    Train the model.

    Args:
        cfg: Hydra config
    """
    # Pull data using DVC if not available locally
    import subprocess
    from pathlib import Path

    data_path = Path(cfg.data.csv_path)
    if not data_path.exists():
        print(f"Data file {data_path} not found locally. Pulling from DVC...")
        try:
            # Use DVC CLI to pull the data
            result = subprocess.run(
                ["uv", "run", "dvc", "pull"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            if result.returncode == 0:
                print("Successfully pulled data from DVC")
            else:
                print(f"Failed to pull data from DVC: {result.stderr}")
                print("Please ensure DVC is properly configured and data is available")
                return
        except Exception as e:
            print(f"Failed to run DVC pull: {e}")
            print("Please ensure DVC is properly configured and data is available")
            return

    # Prepare datamodule kwargs based on model type
    datamodule_kwargs = {
        "csv_path": cfg.data.csv_path,
        "model_type": cfg.model.model_type,
        "batch_size": cfg.data.batch_size,
        "test_size": cfg.data.test_size,
        "val_size": cfg.data.val_size,
        "random_state": cfg.data.random_state,
        "num_workers": cfg.data.num_workers,
    }

    # Add transformer-specific parameters if using transformer model
    if cfg.model.model_type in {"transformer", "small_transformer"}:
        datamodule_kwargs["tokenizer_name"] = cfg.model.tokenizer_name
        datamodule_kwargs["max_length"] = cfg.model.max_length

    datamodule = EmailDataModule(**datamodule_kwargs)

    if cfg.model.model_type in {"transformer", "small_transformer"}:
        # NOTE: we keep the name `train_transformer_model` for compatibility,
        # but it now trains the scratch-based small transformer.
        train_transformer_model(
            datamodule=datamodule,
            model_config=cfg.model,
            learning_rate=cfg.train.learning_rate,
            max_epochs=cfg.train.max_epochs,
            patience=cfg.train.patience,
            mlflow_experiment=cfg.train.mlflow_experiment,
            mlflow_tracking_uri=cfg.logging.mlflow_tracking_uri,
            cpu_threads=cfg.train.cpu_threads,
        )
    elif cfg.model.model_type == "tfidf":
        train_tfidf_model(
            datamodule,
            cfg.train.mlflow_experiment,
            cfg.logging.mlflow_tracking_uri,
        )


def main():
    """
    Main CLI entry point.
    """
    import sys

    # Extract Hydra overrides from command line (e.g., model=baseline)
    # Skip the first argument (script name)
    overrides = sys.argv[1:]

    GlobalHydra.instance().clear()
    config_dir = Path(__file__).parent.parent / "configs"
    initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None)
    cfg = compose(config_name="config", overrides=overrides)
    print(OmegaConf.to_yaml(cfg))

    # For now, just train
    train(cfg)


if __name__ == "__main__":
    # Don't use Fire here - we want to handle args ourselves for Hydra
    main()
