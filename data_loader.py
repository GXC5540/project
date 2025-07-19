import pandas as pd
import jieba
from config import DATA_PATHS


class DataLoader:
    def __init__(self):
        jieba.initialize()

    def load_data(self):
        train_df = pd.read_csv(DATA_PATHS['train'], sep='\t', names=['label', 'content'])
        test_df = pd.read_csv(DATA_PATHS['test'], sep='\t', names=['label', 'content'])
        return train_df, test_df

    def load_stopwords(self):
        with open(DATA_PATHS['stopwords'], encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def preprocess_text(self, text_series):
        return text_series.apply(lambda x: ' '.join(jieba.cut(x)))