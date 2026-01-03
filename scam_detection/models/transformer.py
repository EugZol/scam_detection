"""Transformer models.

This module contains:
1) A HuggingFace-based classifier (kept for backwards compatibility).
2) A lightweight encoder-only Transformer trained from scratch (CPU-friendly).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SmallTransformerConfig:
    """Config for the scratch-trained small transformer."""

    vocab_size: int
    num_labels: int = 2
    max_length: int = 512
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.1


class TransformerClassifier(nn.Module):
    """HuggingFace transformer classifier (pretrained).

    NOTE: This remains for compatibility, but the project default can be switched
    to the scratch-trained model.
    """

    def __init__(
        self, model_name: str = "distilbert-base-uncased", num_labels: int = 2
    ):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class SmallTransformerForSequenceClassification(nn.Module):
    """Encoder-only Transformer classifier trained from scratch.

    Inputs:
        input_ids: LongTensor [B, T]
        attention_mask: LongTensor [B, T] (1 for tokens, 0 for padding)
    Outputs:
        logits: FloatTensor [B, num_labels]
    """

    def __init__(self, cfg: SmallTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_length, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.cfg.max_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_length={self.cfg.max_length}."
            )

        pos_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            # PyTorch expects True for padding positions
            src_key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
            summed = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
