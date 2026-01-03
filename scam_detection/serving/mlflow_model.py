import mlflow
import pandas as pd
import torch
from transformers import AutoTokenizer

from ..models.lit_module import MessageClassifier


class MessageClassifierWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for message classifier.
    """

    def __init__(self, model: MessageClassifier, tokenizer: AutoTokenizer = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, context, model_input):
        """
        Predict method for MLflow.

        Args:
            context: MLflow context
            model_input: Input data

        Returns:
            Predictions
        """
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        else:
            texts = model_input

        # Implement prediction logic based on model type
        if self.model.model_type == "small_transformer":
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for small_transformer model")

            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            # Create batch
            batch = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "label": torch.zeros(len(texts), dtype=torch.long),  # dummy labels
            }

            # Run inference
            with torch.no_grad():
                _, logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)

            return preds.tolist()
        elif self.model.model_type == "tfidf":
            # TF-IDF models don't use tokenizers
            return [0] * len(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model.model_type}")
