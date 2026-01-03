from scam_detection.inference.predictor import Predictor
from scam_detection.models.lit_module import MessageClassifier


def test_predictor():
    model = MessageClassifier(model_type="tfidf")
    predictor = Predictor(model)
    preds = predictor.predict(["test"])
    assert len(preds) == 1
