"""
Smoke test for training components.

Tests basic functionality of data modules and models.
"""

import torch

from scam_detection.data.datamodule import MessageDataModule
from scam_detection.models.lit_module import MessageClassifier


def test_datamodule():
    """Test datamodule setup and data loading."""
    datamodule = MessageDataModule(
        csv_path="tests/fixtures/tiny_dataset.csv",
        model_type="tfidf",
        batch_size=2,
        test_size=0.2,
        val_size=0.2,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # Verify datasets were created
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Test data loading through dataloader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "features" in batch
    assert "label" in batch

    # Verify batch shapes
    assert batch["features"].shape[0] <= 2  # batch_size
    assert batch["label"].shape[0] <= 2

    # Verify label values are valid (0 or 1)
    assert torch.all((batch["label"] == 0) | (batch["label"] == 1))


def test_tfidf_model():
    """Test TF-IDF model creation."""
    model = MessageClassifier(model_type="tfidf")
    assert model is not None
    assert model.model_type == "tfidf"

    # Verify the underlying sklearn model exists
    assert model.model is not None
    assert hasattr(model.model, "fit")
    assert hasattr(model.model, "predict")


def test_transformer_model_forward():
    """Test transformer model creation and forward pass."""
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

    # Create a dummy batch
    batch_size = 2
    seq_length = 10
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones((batch_size, seq_length)),
        "label": torch.tensor([0, 1]),
    }

    # Test forward pass
    model.eval()
    with torch.no_grad():
        loss, logits = model(batch)

    # Verify outputs
    assert loss is not None
    assert logits is not None
    assert logits.shape == (batch_size, 2)  # 2 classes

    # Verify predictions can be extracted
    preds = torch.argmax(logits, dim=1)
    assert preds.shape == (batch_size,)
    assert torch.all((preds == 0) | (preds == 1))


def test_transformer_datamodule():
    """Test datamodule with transformer tokenization."""
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

    # Test data loading
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Verify batch structure for transformer
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch

    # Verify batch shapes
    assert batch["input_ids"].shape[0] <= 2  # batch_size
    assert batch["attention_mask"].shape[0] <= 2
    assert batch["label"].shape[0] <= 2

    # Verify input_ids are valid token IDs (non-negative integers)
    assert torch.all(batch["input_ids"] >= 0)

    # Verify attention_mask contains only 0s and 1s
    assert torch.all((batch["attention_mask"] == 0) | (batch["attention_mask"] == 1))
