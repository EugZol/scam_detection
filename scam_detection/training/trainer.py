from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from ..data.datamodule import MessageDataModule
from ..models.lit_module import MessageClassifier
from ..tracking.mlflow import log_git_commit, setup_mlflow_tracking
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
    import time

    try:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)
        print(f"Configured PyTorch to use {cpu_threads} CPU threads for training")
    except RuntimeError:
        print("PyTorch threads not configured, skipping")

    setup_mlflow_tracking(mlflow_tracking_uri)

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

    run_name = f"small_transformer_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = MLFlowLogger(
        experiment_name=mlflow_experiment,
        run_name=run_name,
        tracking_uri=mlflow_tracking_uri,
    )

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "callbacks": [
            checkpoint_callback,
            early_stopping,
            plotting_callback,
            mlflow_plotting_callback,
        ],
        "logger": logger,
        "log_every_n_steps": log_every_n_steps,
        "accelerator": "cpu",
    }

    if fast_dev_run > 0:
        trainer_kwargs["fast_dev_run"] = fast_dev_run

    trainer = pl.Trainer(**trainer_kwargs)

    datamodule.setup()
    train_samples = len(datamodule.train_dataset)
    val_samples = len(datamodule.val_dataset)

    hyperparams = {
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

    if fast_dev_run == 0:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            if result.returncode == 0:
                hyperparams["git_commit"] = result.stdout.strip()
            else:
                hyperparams["git_commit"] = "unknown"
        except Exception:
            hyperparams["git_commit"] = "unknown"

    trainer.logger.log_hyperparams(hyperparams)

    trainer.fit(model, datamodule)

    trainer.test(model, datamodule)

    if log_model and fast_dev_run == 0:
        import tempfile

        import mlflow.pyfunc
        from transformers import AutoTokenizer

        from ..serving.mlflow_model import MessageClassifierWrapper

        setup_mlflow_tracking(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_id=trainer.logger.run_id):
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "model.ckpt"
                trainer.save_checkpoint(checkpoint_path)

                tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
                tokenizer_path = Path(tmpdir) / "tokenizer"
                tokenizer.save_pretrained(tokenizer_path)

                artifacts = {
                    "model_checkpoint": str(checkpoint_path),
                    "tokenizer": str(tokenizer_path),
                }

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=MessageClassifierWrapper(),
                    artifacts=artifacts,
                    registered_model_name=model_config.model_type,
                )

    return model


def train_tfidf_model(
    datamodule: MessageDataModule,
    model_config,
    mlflow_experiment: str = "message_classification",
    mlflow_tracking_uri: str = "http://127.0.0.1:8080",
    log_model: bool = True,
):
    import time

    import mlflow
    import mlflow.sklearn
    from sklearn.metrics import accuracy_score, f1_score

    setup_mlflow_tracking(mlflow_tracking_uri)

    mlflow.set_experiment(mlflow_experiment)

    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    X_train = []
    y_train = []
    for batch in train_loader:
        X_train.extend(batch["text"])
        y_train.extend(batch["label"].numpy())

    val_loader = datamodule.val_dataloader()
    X_val = []
    y_val = []
    for batch in val_loader:
        X_val.extend(batch["text"])
        y_val.extend(batch["label"].numpy())

    start_time = time.time()
    model = MessageClassifier(
        model_type=model_config.model_type,
        max_features=model_config.max_features,
        stop_words=model_config.stop_words,
        random_state=model_config.random_state,
        max_iter=model_config.max_iter,
    ).model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="binary")

    print(f"TF-IDF Validation Acc: {acc:.4f}, F1: {f1:.4f}")

    with mlflow.start_run(run_name=f"tfidf_baseline_{time.strftime('%Y%m%d_%H%M%S')}"):
        log_git_commit()

        mlflow.set_tag("model_type", "tfidf_baseline")
        mlflow.set_tag("task", "scam_message_detection")
        mlflow.set_tag("framework", "scikit-learn")

        mlflow.log_param("vectorizer_max_features", model_config.max_features)
        mlflow.log_param("vectorizer_stop_words", model_config.stop_words)
        mlflow.log_param("classifier_type", "LogisticRegression")
        mlflow.log_param("random_state", model_config.random_state)
        mlflow.log_param("max_iter", model_config.max_iter)

        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("feature_dim", "auto")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_score", f1)
        mlflow.log_metric("training_time_seconds", training_time)

        if log_model:
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_config.model_type,
            )

    return model
