from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize
from config import MODEL_CONFIG


class self_ComplementNB:
    def __init__(self, alpha=MODEL_CONFIG['alpha'], norm=True):
        self.alpha = alpha
        self.norm = norm
        self.model = ComplementNB(alpha=alpha, norm=norm if norm else None)

    def fit(self, X, y):
        # 如果X是稀疏矩阵且norm=True，需要先归一化
        if self.norm and hasattr(X, "toarray"):
            X = normalize(X, norm='l1', axis=1)
        self.model.fit(X, y)

    def predict(self, X):
        # 同样的归一化处理
        if self.norm and hasattr(X, "toarray"):
            X = normalize(X, norm='l1', axis=1)
        return self.model.predict(X)