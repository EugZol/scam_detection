import fire
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .data.datamodule import EmailDataModule
from .training.trainer import train_transformer_model, train_tfidf_model


def train(cfg: DictConfig):
    """
    Train the model.

    Args:
        cfg: Hydra config
    """
    datamodule = EmailDataModule(
        csv_path=cfg.data.csv_path,
        model_type=cfg.data.model_type,
        tokenizer_name=cfg.data.tokenizer_name,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=cfg.data.random_state
    )

    if cfg.data.model_type == 'transformer':
        train_transformer_model(
            datamodule=datamodule,
            model_name=cfg.model.model_name,
            learning_rate=cfg.train.learning_rate,
            max_epochs=cfg.train.max_epochs,
            patience=cfg.train.patience,
            mlflow_experiment=cfg.train.mlflow_experiment
        )
    elif cfg.data.model_type == 'tfidf':
        train_tfidf_model(datamodule, cfg.train.mlflow_experiment)


def main():
    """
    Main CLI entry point.
    """
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).parent.parent / "configs"
    initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None)
    cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    # For now, just train
    train(cfg)


if __name__ == "__main__":
    fire.Fire(main)
