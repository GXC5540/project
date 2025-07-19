import unittest
import numpy as np
from sklearn.datasets import make_classification
from trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = Trainer()
        # 生成模拟数据
        self.X, self.y = make_classification(n_samples=100, n_features=10)
        self.classes = np.unique(self.y)

    def test_vectorization(self):
        """测试TF-IDF向量化（需适配您的文本数据）"""
        # 此处简化测试，实际应使用文本数据
        X_vec, _, _ = self.trainer.vectorize_text(
                   ["text1", "text2", "text3", "text4"] * 10,
            ["test1", "test2"],
            stopwords=[]
        )
        self.assertEqual(X_vec.shape[0], 40)

    def test_evaluate(self):
        """测试评估指标输出"""
        y_true = [0, 1, 0]
        y_pred = [0, 1, 1]
        self.trainer.evaluate(y_true, y_pred, self.classes, "Test Model")


if __name__ == "__main__":
    unittest.main()