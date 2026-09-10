"""Domain selection using the approved KNN and SetFit models."""

import numpy as np


class DomainClassifier:
    """Average KNN and SetFit class probabilities to select candidate domains."""

    def __init__(self, knn, setfit_model) -> None:
        self.knn = knn
        self.setfit_model = setfit_model

    def classify(self, query: str, query_embedding: np.ndarray, top_n: int) -> dict:
        knn_probabilities = self.knn.predict_proba(query_embedding)[0]
        knn_scores = {str(label): float(score) for label, score in zip(self.knn.classes_, knn_probabilities)}

        setfit_probabilities = self.setfit_model.predict_proba([query])[0]
        setfit_scores = {
            str(label): float(score)
            for label, score in zip(self.setfit_model.labels, setfit_probabilities)
        }
        domains = set(knn_scores) | set(setfit_scores)
        combined = {
            domain: 0.5 * knn_scores.get(domain, 0.0) + 0.5 * setfit_scores.get(domain, 0.0)
            for domain in domains
        }
        combined = dict(sorted(combined.items(), key=lambda item: item[1], reverse=True))
        return {
            "knn_scores": knn_scores,
            "setfit_scores": setfit_scores,
            "combined_scores": combined,
            "domains": list(combined)[:top_n],
        }
