import pandas as pd
import lightning.pytorch as pl
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import EmailDataset, TfidfEmailDataset
from .preprocessing import load_and_preprocess_data, prepare_tfidf_features


class EmailDataModule(pl.LightningDataModule):
    """DataModule for email classification."""

    def __init__(
        self,
        csv_path: str,
        model_type: str = 'transformer',
        tokenizer_name: str = 'distilbert-base-uncased',
        max_length: int = 512,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        super().__init__()
        self.csv_path = csv_path
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.tokenizer = None
        self.vectorizer = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = load_and_preprocess_data(self.csv_path)

        train_val_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state, stratify=df['label']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=self.val_size, random_state=self.random_state, stratify=train_val_df['label']
        )

        if self.model_type == 'transformer':
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.train_dataset = EmailDataset(
                train_df['text'].tolist(), train_df['label'].tolist(), self.tokenizer, self.max_length
            )
            self.val_dataset = EmailDataset(
                val_df['text'].tolist(), val_df['label'].tolist(), self.tokenizer, self.max_length
            )
            self.test_dataset = EmailDataset(
                test_df['text'].tolist(), test_df['label'].tolist(), self.tokenizer, self.max_length
            )
        elif self.model_type == 'tfidf':
            self.vectorizer = prepare_tfidf_features(train_df['text'].tolist())

            train_features = torch.tensor(self.vectorizer.transform(train_df['text']).toarray(), dtype=torch.float32)
            val_features = torch.tensor(self.vectorizer.transform(val_df['text']).toarray(), dtype=torch.float32)
            test_features = torch.tensor(self.vectorizer.transform(test_df['text']).toarray(), dtype=torch.float32)

            self.train_dataset = TfidfEmailDataset(train_features, train_df['label'].tolist())
            self.val_dataset = TfidfEmailDataset(val_features, val_df['label'].tolist())
            self.test_dataset = TfidfEmailDataset(test_features, test_df['label'].tolist())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
