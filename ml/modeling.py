"""Shared, serializable inference interface for raw and calibrated classifiers."""

from dataclasses import dataclass

from sklearn.feature_extraction import DictVectorizer


@dataclass
class MatchModel:
    vectorizer: DictVectorizer
    estimator: object
    name: str
    draft: bool
    calibrated: bool

    def predict_proba(self, features):
        return self.estimator.predict_proba(self.vectorizer.transform(features))
