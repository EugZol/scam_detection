#!/usr/bin/env python3
"""
Smoke test for inference.
"""

from scam_detection.inference.predictor import Predictor
from scam_detection.models.lit_module import EmailClassifier


def smoke_test():
    """
    Run smoke test for inference.
    """
    # Load model (mock)
    model = EmailClassifier(model_type='transformer')  # This will load default

    predictor = Predictor(model)

    test_texts = ["This is a safe email.", "This is a phishing email!"]

    preds = predictor.predict(test_texts)

    print(f"Predictions: {preds}")


if __name__ == "__main__":
    smoke_test()
