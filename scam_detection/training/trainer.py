import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from ..data.datamodule import MessageDataModule
from ..models.lit_module import MessageClassifier
from ..tracking.mlflow import (
    cleanup_malformed_experiments,
    ensure_experiment_exists,
    log_git_commit,
    setup_mlflow_tracking,
)
from .callbacks import MLflowPlottingCallback, PlottingCallback


def train_transformer_model(
    datamodule: MessageDataModule,
    model_config,
    learning_rate: float = 2e-5,
    max_epochs: int = 10,
    patience: int = 3,
    mlflow_experiment: str = "message_classification",
    mlflow_tracking_uri: str = "http://127.0.0.1:8080",
    log_every_n_steps: int = 20,
    cpu_threads: int = 20,
    fast_dev_run: int = 0,
    log_model: bool = True,
):
    """Train the transformer model."""
    import time

    # Configure PyTorch for multi-core CPU training
    try:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)
        print(f"Configured PyTorch to use {cpu_threads} CPU threads for training")
    except RuntimeError:
        print("PyTorch threads already configured, skipping")

    # Set up MLflow tracking
    setup_mlflow_tracking(mlflow_tracking_uri)
    cleanup_malformed_experiments()
    ensure_experiment_exists(mlflow_experiment)

    # Set the MLflow experiment explicitly
    import mlflow

    mlflow.set_experiment(mlflow_experiment)

    model = MessageClassifier(
        model_type=model_config.model_type,
        tokenizer_name=model_config.tokenizer_name,
        max_length=model_config.max_length,
        small_d_model=model_config.d_model,
        small_n_heads=model_config.n_heads,
        small_n_layers=model_config.n_layers,
        small_ffn_dim=model_config.ffn_dim,
        small_dropout=model_config.dropout,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename=f"{model_config.model_type}-{{epoch:02d}}-{{val_f1:.2f}}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_f1", patience=patience, mode="max")
    plotting_callback = PlottingCallback()
    mlflow_plotting_callback = MLflowPlottingCallback(
        log_every_n_steps=log_every_n_steps
    )

    # Enhanced MLflow logger with descriptive run name
    run_name = f"small_transformer_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = MLFlowLogger(experiment_name=mlflow_experiment, run_name=run_name)

    # Build trainer kwargs
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "callbacks": [
            checkpoint_callback,
            early_stopping,
            plotting_callback,
            mlflow_plotting_callback,
        ],
        "logger": logger,
        "log_every_n_steps": 1,
        "accelerator": "cpu",
    }

    # Add fast_dev_run if specified (for quick testing)
    if fast_dev_run > 0:
        trainer_kwargs["fast_dev_run"] = fast_dev_run

    trainer = pl.Trainer(**trainer_kwargs)

    # Log additional metadata before training
    datamodule.setup()
    train_samples = len(datamodule.train_dataset)
    val_samples = len(datamodule.val_dataset)
    trainer.logger.log_hyperparams(
        {
            "model_type": model_config.model_type,
            "tokenizer_name": model_config.tokenizer_name,
            "task": "scam_message_detection",
            "framework": "pytorch_lightning",
            "train_samples": train_samples,
            "val_samples": val_samples,
            "max_length": model_config.max_length,
            "batch_size": datamodule.batch_size,
            "num_workers": datamodule.num_workers,
            "cpu_threads": cpu_threads,
            "d_model": model_config.d_model,
            "n_heads": model_config.n_heads,
            "n_layers": model_config.n_layers,
            "ffn_dim": model_config.ffn_dim,
            "dropout": model_config.dropout,
        }
    )

    # Log git commit before training starts (when MLflow run is active)
    # Skip if fast_dev_run is enabled (logging is suppressed in that mode)
    if fast_dev_run == 0:
        log_git_commit()

    trainer.fit(model, datamodule)

    # Log final model checkpoint (best ckpt path) to MLflow artifacts
    if checkpoint_callback.best_model_path:
        trainer.logger.experiment.log_artifact(
            trainer.logger.run_id,
            checkpoint_callback.best_model_path,
            artifact_path="best_model_checkpoint",
        )

    trainer.test(model, datamodule)

    # Log model to MLflow (can be slow, so optional)
    if log_model and fast_dev_run == 0:
        import mlflow.pytorch

        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_config.model_type,
        )

    return model


def train_tfidf_model(
    datamodule: MessageDataModule,
    model_type: str,
    mlflow_experiment: str = "message_classification",
    mlflow_tracking_uri: str = "http://127.0.0.1:8080",
    log_model: bool = True,
):
    """Train the TF-IDF baseline model.

    Args:
        log_model: Whether to log model to MLflow (can be slow, disable in tests).
        model_type: Model type identifier used for registered model name.
    """
    import time

    import mlflow
    import mlflow.sklearn
    from sklearn.metrics import accuracy_score, f1_score

    # Set up MLflow tracking
    setup_mlflow_tracking(mlflow_tracking_uri)

    # Set the MLflow experiment (creates if doesn't exist)
    mlflow.set_experiment(mlflow_experiment)

    datamodule.setup()

    # Collect training data
    train_loader = datamodule.train_dataloader()
    X_train = []
    y_train = []
    for batch in train_loader:
        X_train.extend(batch["features"].numpy())
        y_train.extend(batch["label"].numpy())

    # Collect validation data
    val_loader = datamodule.val_dataloader()
    X_val = []
    y_val = []
    for batch in val_loader:
        X_val.extend(batch["features"].numpy())
        y_val.extend(batch["label"].numpy())

    # Train model
    start_time = time.time()
    model = MessageClassifier(model_type="tfidf").model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="binary")

    print(f"TF-IDF Validation Acc: {acc:.4f}, F1: {f1:.4f}")

    # Log to MLflow with detailed metadata
    with mlflow.start_run(run_name=f"tfidf_baseline_{time.strftime('%Y%m%d_%H%M%S')}"):
        # Log git commit
        log_git_commit()

        # Log model details
        mlflow.set_tag("model_type", "tfidf_baseline")
        mlflow.set_tag("task", "scam_message_detection")
        mlflow.set_tag("framework", "scikit-learn")

        # Log hyperparameters
        mlflow.log_param(
            "vectorizer_max_features", getattr(model, "max_features", "default")
        )
        mlflow.log_param("vectorizer_stop_words", "english")
        mlflow.log_param("classifier_type", "LogisticRegression")

        # Log data info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("feature_dim", X_train[0].shape[0] if X_train else 0)

        # Log metrics
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_score", f1)
        mlflow.log_metric("training_time_seconds", training_time)

        # Log model to MLflow (can be slow, so optional)
        if log_model:
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_type,
            )

    return model
