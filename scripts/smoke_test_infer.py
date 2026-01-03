#!/usr/bin/env python3
"""
Smoke test for inference.
"""

from scam_detection.inference.predictor import Predictor
from scam_detection.models.lit_module import MessageClassifier


def smoke_test():
    """
    Run smoke test for inference.
    """
    from transformers import AutoTokenizer

    # Load model (mock)
    model = MessageClassifier(model_type="transformer")  # This will load default
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    predictor = Predictor(model, tokenizer)

    test_texts = ["This is a safe message.", "This is a scam message!"]

    preds = predictor.predict(test_texts)

    print(f"Predictions: {preds}")


if __name__ == "__main__":
    smoke_test()
