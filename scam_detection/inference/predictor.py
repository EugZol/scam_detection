from typing import List

import torch
from transformers import AutoTokenizer

from ..models.lit_module import MessageClassifier


class Predictor:
    def __init__(self, model: MessageClassifier, tokenizer: AutoTokenizer = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, texts: List[str]) -> List[int]:
        if self.model.model_type == "small_transformer":
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "label": torch.zeros(len(texts), dtype=torch.long),
            }
            with torch.no_grad():
                _, logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)
            return preds.tolist()
        elif self.model.model_type == "tfidf":
            return [0] * len(texts)
