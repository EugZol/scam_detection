"""
Smoke test for training.
"""

import pytest

from scam_detection.data.datamodule import EmailDataModule
from scam_detection.models.lit_module import EmailClassifier


def test_datamodule():
    datamodule = EmailDataModule(
        csv_path="data/Phishing_Email.csv",
        model_type="tfidf",
        batch_size=2
    )
    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.train_dataset is not None


def test_model():
    model = EmailClassifier(model_type="tfidf")
    assert model is not None
