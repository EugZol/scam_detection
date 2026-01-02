from sklearn.linear_model import LogisticRegression


class TfidfClassifier:
    """Baseline classifier using TF-IDF features and Logistic Regression."""

    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)

    def fit(self, X_train, y_train):
        """Fit the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
