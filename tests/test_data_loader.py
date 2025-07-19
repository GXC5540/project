import unittest
import pandas as pd
from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.train_df, cls.test_df = cls.loader.load_data()

    def test_data_loading(self):
        self.assertIsInstance(self.train_df, pd.DataFrame)
        self.assertIsInstance(self.test_df, pd.DataFrame)
        self.assertEqual(len(self.train_df), 50000)  # 假设训练集有5万条

    def test_preprocess_text(self):
        sample_text = pd.Series(["自然语言处理很有趣"])
        processed = self.loader.preprocess_text(sample_text)
        self.assertEqual(processed[0], "自然语言 处理 很 有趣")  # 检查分词结果


if __name__ == "__main__":
    unittest.main()