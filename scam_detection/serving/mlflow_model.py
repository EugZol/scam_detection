import mlflow
import pandas as pd
import torch
from transformers import AutoTokenizer

from ..models.lit_module import MessageClassifier


class MessageClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model: MessageClassifier, tokenizer: AutoTokenizer = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        else:
            texts = model_input

        if self.model.model_type == "small_transformer":
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for small_transformer model")

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
                "label": torch.zeros(len(texts), dtype=torch.long),  # dummy labels
            }

            with torch.no_grad():
                _, logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)

            return preds.tolist()
        elif self.model.model_type == "tfidf":
            return self.model.predict(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model.model_type}")
