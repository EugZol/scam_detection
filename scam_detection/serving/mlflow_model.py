import os

import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class MessageClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle

        checkpoint_path = context.artifacts["model_checkpoint"]
        from ..models.lit_module import MessageClassifier

        self.model = MessageClassifier.load_from_checkpoint(checkpoint_path)
        self.model.eval()

        if "tokenizer" in context.artifacts:
            tokenizer_path = context.artifacts["tokenizer"]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, np.ndarray):
            if model_input.ndim == 1:
                texts = model_input.astype(str).tolist()
            elif model_input.ndim == 2 and model_input.shape[1] >= 1:
                texts = model_input[:, 0].astype(str).tolist()
            else:
                raise ValueError(
                    f"Unsupported ndarray shape: {model_input.shape}. "
                    "Expected (n,) or (n,1)."
                )
        elif isinstance(model_input, (list, tuple, pd.Series)):
            texts = [str(x) for x in list(model_input)]
        elif isinstance(model_input, dict) and "inputs" in model_input:
            texts = [str(x) for x in model_input["inputs"]]
        else:
            raise ValueError(
                f"Unsupported input type: {type(model_input)}. "
                "Expected DataFrame, ndarray, list/tuple/Series, or dict with 'inputs' key."
            )

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
                "label": torch.zeros(len(texts), dtype=torch.long),
            }

            with torch.no_grad():
                _, logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)

            return preds.tolist()
        elif self.model.model_type == "tfidf":
            return self.model.predict(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model.model_type}")
