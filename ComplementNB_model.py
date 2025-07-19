import numpy as np
from collections import defaultdict
import math
from config import MODEL_CONFIG


class ComplementNB:
    def __init__(self, alpha=MODEL_CONFIG['alpha'], norm=True):
        self.alpha = alpha
        self.norm = norm
        self.class_prior = None
        self.feature_log_prob = None
        self.classes = None
        self.n_features = None
        self.feature_all = None  # for storing feature counts across all classes

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.class_prior = {}
        class_counts = defaultdict(int)

        # Compute class prior probabilities
        for cls in self.classes:
            class_counts[cls] = np.sum(y == cls)
            self.class_prior[cls] = math.log(class_counts[cls] / len(y))

        # Compute complement feature counts
        self.feature_all = np.zeros(self.n_features)
        feature_counts = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            counts = np.array(X_cls.sum(axis=0))[0] if hasattr(X_cls, "toarray") else np.sum(X_cls, axis=0)
            feature_counts[cls] = counts
            self.feature_all += counts

        # Compute complement feature log probabilities
        self.feature_log_prob = {}
        for cls in self.classes:
            # Complement counts: sum of features in all classes except this one
            complement_counts = self.feature_all - feature_counts[cls]
            total_complement = np.sum(complement_counts)

            # Calculate probabilities with smoothing
            probs = (complement_counts + self.alpha) / (total_complement + self.alpha * self.n_features)

            # Take log and optionally normalize by class counts
            if self.norm:
                probs /= class_counts[cls]

            self.feature_log_prob[cls] = np.log(probs)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        x = x.toarray()[0] if hasattr(x, "toarray") else x
        log_probs = {
            cls: self.class_prior[cls] - np.sum(x[np.where(x > 0)] * self.feature_log_prob[cls][np.where(x > 0)])
            for cls in self.classes
        }
        return max(log_probs.items(), key=lambda x: x[1])[0]