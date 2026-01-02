import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for email classification."""

    def __init__(
        self, model_name: str = "distilbert-base-uncased", num_labels: int = 2
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
