import torch

from scam_detection.data.datamodule import MessageDataModule
from scam_detection.models.lit_module import MessageClassifier


def test_datamodule():
    datamodule = MessageDataModule(
        csv_path="tests/fixtures/tiny_dataset.csv",
        model_type="tfidf",
        batch_size=2,
        test_size=0.2,
        val_size=0.2,
    )
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    assert "text" in batch
    assert "label" in batch

    assert len(batch["text"]) <= 2
    assert batch["label"].shape[0] <= 2

    assert torch.all((batch["label"] == 0) | (batch["label"] == 1))


def test_tfidf_model():
    model = MessageClassifier(model_type="tfidf")
    assert model is not None
    assert model.model_type == "tfidf"

    assert model.model is not None
    assert hasattr(model.model, "fit")
    assert hasattr(model.model, "predict")


def test_transformer_model_forward():
    model = MessageClassifier(
        model_type="small_transformer",
        tokenizer_name="bert-base-uncased",
        small_d_model=128,
        small_n_heads=2,
        small_n_layers=2,
        small_ffn_dim=256,
        max_length=128,
    )
    assert model is not None
    assert model.model_type == "small_transformer"

    batch_size = 2
    seq_length = 10
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones((batch_size, seq_length)),
        "label": torch.tensor([0, 1]),
    }

    model.eval()
    with torch.no_grad():
        loss, logits = model(batch)

    assert loss is not None
    assert logits is not None
    assert logits.shape == (batch_size, 2)

    preds = torch.argmax(logits, dim=1)
    assert preds.shape == (batch_size,)
    assert torch.all((preds == 0) | (preds == 1))


def test_transformer_datamodule():
    datamodule = MessageDataModule(
        csv_path="tests/fixtures/tiny_dataset.csv",
        model_type="small_transformer",
        tokenizer_name="bert-base-uncased",
        max_length=128,
        batch_size=2,
        test_size=0.2,
        val_size=0.2,
    )
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch

    assert batch["input_ids"].shape[0] <= 2
    assert batch["attention_mask"].shape[0] <= 2
    assert batch["label"].shape[0] <= 2

    assert torch.all(batch["input_ids"] >= 0)

    assert torch.all((batch["attention_mask"] == 0) | (batch["attention_mask"] == 1))
