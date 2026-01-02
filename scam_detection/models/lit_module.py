import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .baseline import TfidfClassifier
from .transformer import TransformerClassifier


class EmailClassifier(pl.LightningModule):
    """Lightning module for email classification."""

    def __init__(
        self,
        model_type: str = 'transformer',
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_epochs: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        if model_type == 'transformer':
            self.model = TransformerClassifier(model_name, num_labels)
        elif model_type == 'tfidf':
            self.model = TfidfClassifier()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        if self.model_type == 'transformer':
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label']
            )
            return outputs.loss, outputs.logits
        elif self.model_type == 'tfidf':
            return None, self.model.predict_proba(batch['features'])

    def training_step(self, batch, batch_idx):
        if self.model_type == 'transformer':
            loss, logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(batch['label'].cpu(), preds.cpu())
            f1 = f1_score(batch['label'].cpu(), preds.cpu(), average='binary')
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        elif self.model_type == 'tfidf':
            return torch.tensor(0.0)

    def validation_step(self, batch, batch_idx):
        if self.model_type == 'transformer':
            loss, logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(batch['label'].cpu(), preds.cpu())
            f1 = f1_score(batch['label'].cpu(), preds.cpu(), average='binary')
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        if self.model_type == 'transformer':
            loss, logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(batch['label'].cpu(), preds.cpu())
            f1 = f1_score(batch['label'].cpu(), preds.cpu(), average='binary')
            self.log('test_acc', acc)
            self.log('test_f1', f1)

    def configure_optimizers(self):
        if self.model_type == 'transformer':
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [scheduler]
        return None
