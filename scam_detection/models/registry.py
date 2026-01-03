from typing import Dict, Type

from .baseline import TfidfClassifier
from .lit_module import MessageClassifier
from .transformer import TransformerClassifier

MODEL_REGISTRY: Dict[str, Type] = {
    "tfidf": TfidfClassifier,
    "transformer": TransformerClassifier,
    "message_classifier": MessageClassifier,
}
