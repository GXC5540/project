import numpy as np
from data_loader import DataLoader
from trainer import Trainer
from config import MODEL_CONFIG
import pickle
import os


def main(model):
    # 初始化组件
    loader = DataLoader()
    trainer = Trainer()

    # 数据加载与预处理
    train_df, test_df = loader.load_data()
    stopwords = loader.load_stopwords()
    X_train = loader.preprocess_text(train_df['content'])
    y_train = train_df['label']
    X_test = loader.preprocess_text(test_df['content'])
    y_test = test_df['label']
    classes = np.unique(y_train)

    # 特征工程
    X_train_vec, X_test_vec, tfidf = trainer.vectorize_text(X_train, X_test, stopwords)
    X_train_sel, X_test_sel, selector = trainer.select_features(X_train_vec, y_train, X_test_vec)

    models = {
        'Full Features': model(alpha=MODEL_CONFIG['alpha']),
        'Selected Features': model(alpha=MODEL_CONFIG['alpha'])
    }

    models['Full Features'].fit(X_train_vec, y_train)
    models['Selected Features'].fit(X_train_sel, y_train)

    # 保存 Selected Features 模型
    save_dir = "D:\\pycharm2\\pythonProject2\\BYS"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    # 保存模型、TF-IDF 向量化器和特征选择器
    with open(os.path.join(save_dir, 'selected_model.pkl'), 'wb') as f:
        pickle.dump({
            'model': models['Selected Features'],
            'tfidf': tfidf,
            'selector': selector
        }, f)
    print(f"模型已保存到 {save_dir}")

    for name, model in models.items():
        y_pred = model.predict(X_test_sel if 'Selected' in name else X_test_vec)
        trainer.evaluate(y_test, y_pred, classes, name)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
