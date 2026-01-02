"""
Smoke test for inference.
"""

from scam_detection.inference.predictor import Predictor
from scam_detection.models.lit_module import EmailClassifier


def test_predictor():
    model = EmailClassifier(model_type="tfidf")
    predictor = Predictor(model)
    preds = predictor.predict(["test"])
    assert len(preds) == 1
