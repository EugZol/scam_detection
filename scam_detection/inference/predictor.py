from typing import List

import torch
from transformers import AutoTokenizer

from ..models.lit_module import EmailClassifier


class Predictor:
    """Predictor for email classification."""

    def __init__(self, model: EmailClassifier, tokenizer: AutoTokenizer = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, texts: List[str]) -> List[int]:
        """Predict labels for texts."""
        if self.model.model_type == 'transformer':
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
            return preds.tolist()
        elif self.model.model_type == 'tfidf':
            return [0] * len(texts)
