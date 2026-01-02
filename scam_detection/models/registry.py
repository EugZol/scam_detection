from typing import Dict, Type

from .baseline import TfidfClassifier
from .lit_module import EmailClassifier
from .transformer import TransformerClassifier

MODEL_REGISTRY: Dict[str, Type] = {
    "tfidf": TfidfClassifier,
    "transformer": TransformerClassifier,
    "email_classifier": EmailClassifier,
}
