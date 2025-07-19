# 路径配置
DATA_PATHS = {
    'train': "C:\\Users\\HUAWEI\\Desktop\\cnews.train.txt",
    'test': "C:\\Users\\HUAWEI\\Desktop\\cnews.test.txt",
    'stopwords': "C:\\Users\\HUAWEI\\Desktop\\cnews.vocab.txt"
}

# 模型参数
MODEL_CONFIG = {
    'alpha': 1.0,
    'max_features': 15000,
    'k_best': 12000,
    'max_df': 0.95,
    'min_df': 1,
    'ngram_range': (1, 3),
}

# 可视化配置
PLOT_CONFIG = {
    'font': 'SimHei',
    'figsize': (10, 8),
    'cmap': 'Blues'
}