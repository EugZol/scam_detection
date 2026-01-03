import lightning.pytorch as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from transformers import get_linear_schedule_with_warmup

from .baseline import TfidfClassifier
from .transformer import (
    SmallTransformerConfig,
    SmallTransformerForSequenceClassification,
    TransformerClassifier,
)


class MessageClassifier(pl.LightningModule):
    """Lightning module for message classification."""

    def __init__(
        self,
        model_type: str = "transformer",
        model_name: str = "distilbert-base-uncased",
        tokenizer_name: str = "bert-base-uncased",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_epochs: int = 10,
        # small transformer params
        small_d_model: int = 384,
        small_n_heads: int = 6,
        small_n_layers: int = 6,
        small_ffn_dim: int = 1536,
        small_dropout: float = 0.1,
        max_length: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        if model_type == "transformer":
            # Pretrained HuggingFace model (legacy option)
            self.model = TransformerClassifier(model_name, num_labels)
        elif model_type == "small_transformer":
            # Scratch-trained transformer encoder
            from transformers import AutoTokenizer

            vocab_size = AutoTokenizer.from_pretrained(tokenizer_name).vocab_size
            cfg = SmallTransformerConfig(
                vocab_size=vocab_size,
                num_labels=num_labels,
                max_length=max_length,
                d_model=small_d_model,
                n_heads=small_n_heads,
                n_layers=small_n_layers,
                ffn_dim=small_ffn_dim,
                dropout=small_dropout,
            )
            self.model = SmallTransformerForSequenceClassification(cfg)
        elif model_type == "tfidf":
            self.model = TfidfClassifier()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.criterion = nn.CrossEntropyLoss()

        # Initialize torchmetrics for proper validation/test metric calculation
        # These automatically accumulate across batches and reset each epoch
        # Use 'multiclass' task since we pass class indices (from argmax),
        # not probabilities
        if model_type in {"transformer", "small_transformer"}:
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
            self.val_f1 = F1Score(
                task="multiclass",
                num_classes=num_labels,
                average="macro",
            )
            self.test_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
            self.test_f1 = F1Score(
                task="multiclass",
                num_classes=num_labels,
                average="macro",
            )

    def forward(self, batch):
        if self.model_type == "transformer":
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            return outputs.loss, outputs.logits
        if self.model_type == "small_transformer":
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            loss = self.criterion(logits, batch["label"])
            return loss, logits
        elif self.model_type == "tfidf":
            return None, self.model.predict_proba(batch["features"])

    def training_step(self, batch, batch_idx):
        if self.model_type in {"transformer", "small_transformer"}:
            loss, logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(batch["label"].cpu(), preds.cpu())
            f1 = f1_score(batch["label"].cpu(), preds.cpu(), average="binary")
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_acc",
                acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )
            return loss
        elif self.model_type == "tfidf":
            return torch.tensor(0.0)

    def validation_step(self, batch, batch_idx):
        if self.model_type in {"transformer", "small_transformer"}:
            loss, logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)

            # Update torchmetrics - they accumulate across all batches automatically
            self.val_accuracy.update(preds, batch["label"])
            self.val_f1.update(preds, batch["label"])

            # Log loss (average across batches is fine for loss)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            # Log metrics - torchmetrics will compute the final value at epoch end
            self.log(
                "val_acc",
                self.val_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1",
                self.val_f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        if self.model_type in {"transformer", "small_transformer"}:
            loss, logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)

            # Update torchmetrics - they accumulate across all batches automatically
            self.test_accuracy.update(preds, batch["label"])
            self.test_f1.update(preds, batch["label"])

            # Log metrics - torchmetrics will compute the final value at epoch end
            self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)
            self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.model_type in {"transformer", "small_transformer"}:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [scheduler]
        return None
