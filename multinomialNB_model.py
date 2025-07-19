import numpy as np
from collections import defaultdict
import math
from config import MODEL_CONFIG


class MultinomialNB:
    def __init__(self, alpha=MODEL_CONFIG['alpha']):
        self.alpha = alpha
        self.class_prior = None
        self.feature_prob = None
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.class_prior = {}
        class_counts = defaultdict(int)

        # 计算类先验概率
        for cls in self.classes:
            class_counts[cls] = np.sum(y == cls)
            self.class_prior[cls] = math.log(class_counts[cls] / len(y))

        # 计算特征条件概率
        self.feature_prob = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            feature_counts = np.array(X_cls.sum(axis=0))[0] if hasattr(X_cls, "toarray") else np.sum(X_cls, axis=0)
            total_count = np.sum(feature_counts)
            self.feature_prob[cls] = np.log(
                (feature_counts + self.alpha) / (total_count + self.alpha * self.n_features))

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        x = x.toarray()[0] if hasattr(x, "toarray") else x
        log_probs = {
            cls: self.class_prior[cls] + np.sum(x[np.where(x > 0)] * self.feature_prob[cls][np.where(x > 0)])
            for cls in self.classes
        }
        return max(log_probs.items(), key=lambda x: x[1])[0]