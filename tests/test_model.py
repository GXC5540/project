import unittest
import numpy as np
from multinomialNB_model import MultinomialNB


class TestMultinomialNB(unittest.TestCase):
    def setUp(self):
        # 模拟训练数据
        self.X_train = np.array([
            [1, 2, 0],
            [0, 3, 1],
            [1, 0, 2]
        ])
        self.y_train = np.array(["A", "B", "A"])
        self.clf = MultinomialNB(alpha=0.1)
        self.clf.fit(self.X_train, self.y_train)

    def test_predict(self):
        X_test = np.array([[1, 1, 0]])
        pred = self.clf.predict(X_test)
        self.assertEqual(pred[0], "A")  # 预期类别A

    def test_class_prior(self):
        self.assertAlmostEqual(
            np.exp(self.clf.class_prior["A"]),
            2/3,
            delta=0.01
        )


if __name__ == "__main__":
    unittest.main()