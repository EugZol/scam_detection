import re
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["Message Text", "Message Type"]]
    df.columns = ["text", "label"]
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].map({"Safe Message": 0, "Scam Message": 1})
    return df.dropna()


def prepare_tfidf_features(
    texts: List[str],
    vectorizer: TfidfVectorizer = None,
    fit: bool = True,
    max_features: int = 5000,
    stop_words: str = "english"
) -> TfidfVectorizer:
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    if fit:
        vectorizer.fit(texts)
    return vectorizer


def prepare_transformer_features(
    texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512
) -> dict:
    return tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
