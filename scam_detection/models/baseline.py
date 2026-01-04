from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfClassifier:
    def __init__(
        self,
        max_features: int = 5000,
        stop_words: str = 'english',
        random_state: int = 42,
        max_iter: int = 1000
    ):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words=stop_words)),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=max_iter))
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
