import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from ..data.datamodule import EmailDataModule
from ..models.lit_module import EmailClassifier
from ..tracking.mlflow import log_git_commit, setup_mlflow_tracking
from .callbacks import PlottingCallback


def train_transformer_model(
    datamodule: EmailDataModule,
    model_name: str = "distilbert-base-uncased",
    learning_rate: float = 2e-5,
    max_epochs: int = 10,
    patience: int = 3,
    mlflow_experiment: str = "email_classification",
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """Train the transformer model."""
    import time

    # Set up MLflow tracking
    setup_mlflow_tracking(mlflow_tracking_uri)

    model = EmailClassifier(
        model_type="transformer",
        model_name=model_name,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="transformer-{epoch:02d}-{val_f1:.2f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_f1", patience=patience, mode="max")
    plotting_callback = PlottingCallback()

    # Enhanced MLflow logger with descriptive run name
    run_name = (
        f"transformer_{model_name.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    logger = MLFlowLogger(
        experiment_name=mlflow_experiment, run_name=run_name, log_model=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping, plotting_callback],
        logger=logger,
        # Make sure metrics show up as a time-series in MLflow quickly.
        # (otherwise you may only see epoch-level aggregation)
        log_every_n_steps=1,
        accelerator="cpu",
    )

    # Log additional metadata before training
    datamodule.setup()
    train_samples = len(datamodule.train_dataset)
    val_samples = len(datamodule.val_dataset)

    trainer.logger.log_hyperparams(
        {
            "model_type": "transformer",
            "model_name": model_name,
            "task": "email_phishing_detection",
            "framework": "pytorch_lightning",
            "train_samples": train_samples,
            "val_samples": val_samples,
            "max_length": datamodule.max_length,
            "batch_size": datamodule.batch_size,
        }
    )

    trainer.fit(model, datamodule)

    # Log git commit after training starts (when MLflow run is active)
    log_git_commit()

    # Log final model checkpoint
    trainer.logger.log_artifact(
        checkpoint_callback.best_model_path, "best_model_checkpoint"
    )

    trainer.test(model, datamodule)

    return model


def train_tfidf_model(
    datamodule: EmailDataModule, mlflow_experiment: str = "email_classification"
):
    """Train the TF-IDF baseline model."""
    import time

    import mlflow
    import mlflow.sklearn
    from sklearn.metrics import accuracy_score, f1_score

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
    model = EmailClassifier(model_type="tfidf").model
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
        mlflow.set_tag("task", "email_phishing_detection")
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

        # Log model with descriptive name
        mlflow.sklearn.log_model(
            model,
            "tfidf_logistic_regression_model",
            registered_model_name="EmailPhishingDetector_TFIDF",
        )

    return model
