from typing import List

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmailDataset(Dataset):
    """Dataset for email classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer = None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }


class TfidfEmailDataset(Dataset):
    """Dataset for TF-IDF features."""

    def __init__(self, features: torch.Tensor, labels: List[int]):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
