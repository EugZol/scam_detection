from scam_detection.inference.predictor import Predictor
from scam_detection.models.lit_module import MessageClassifier


def test_predictor():
    model = MessageClassifier(model_type="tfidf")

    train_texts = ["this is a test message", "another sample text", "spam offer"]
    train_labels = [0, 0, 1]
    model.model.fit(train_texts, train_labels)

    predictor = Predictor(model)
    preds = predictor.predict(["test"])
    assert len(preds) == 1
