import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from ..data.datamodule import EmailDataModule
from ..models.lit_module import EmailClassifier


def train_transformer_model(
    datamodule: EmailDataModule,
    model_name: str = 'distilbert-base-uncased',
    learning_rate: float = 2e-5,
    max_epochs: int = 10,
    patience: int = 3,
    mlflow_experiment: str = 'email_classification'
):
    """Train the transformer model."""
    model = EmailClassifier(
        model_type='transformer',
        model_name=model_name,
        learning_rate=learning_rate,
        max_epochs=max_epochs
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='transformer-{epoch:02d}-{val_f1:.2f}',
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )
    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=patience,
        mode='max'
    )

    logger = MLFlowLogger(experiment_name=mlflow_experiment)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        # Make sure metrics show up as a time-series in MLflow quickly.
        # (otherwise you may only see epoch-level aggregation)
        log_every_n_steps=1,
        accelerator='auto',
        devices='auto'
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    return model


def train_tfidf_model(datamodule: EmailDataModule, mlflow_experiment: str = 'email_classification'):
    """Train the TF-IDF baseline model."""
    from sklearn.metrics import accuracy_score, f1_score
    import mlflow
    import mlflow.sklearn

    datamodule.setup()
    vectorizer = datamodule.vectorizer

    train_loader = datamodule.train_dataloader()
    X_train = []
    y_train = []
    for batch in train_loader:
        X_train.extend(batch['features'].numpy())
        y_train.extend(batch['label'].numpy())

    model = EmailClassifier(model_type='tfidf').model
    model.fit(X_train, y_train)

    val_loader = datamodule.val_dataloader()
    X_val = []
    y_val = []
    for batch in val_loader:
        X_val.extend(batch['features'].numpy())
        y_val.extend(batch['label'].numpy())

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='binary')

    print(f"TF-IDF Validation Acc: {acc:.4f}, F1: {f1:.4f}")

    with mlflow.start_run(experiment_id=mlflow_experiment):
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_metric('val_acc', acc)
        mlflow.log_metric('val_f1', f1)

    return model
